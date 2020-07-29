 /*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed
 * with this work for additional information regarding copyright
 * ownership.  The ASF licenses this file to you under the Apache
 * License, Version 2.0 (the "License"); you may not use this file
 * except in compliance with the License.  You may obtain a copy of
 * the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file cudnn_norm_convolution-inl.h
 * \brief
 * \author Dick Carter
*/
#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_NORM_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_NORM_CONVOLUTION_INL_H_

#include <mxnet/storage.h>
#include <algorithm>
#include <vector>
#include <mutex>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include "../convolution-inl.h"
#include "../norm_convolution-inl.h"
#include "../../../common/cuda_utils.h"
#include "nhwc_batch_norm-inl.h"
#include "cudnn_common_op.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1

/*!
 * \brief The Operator used to perform normalized convolution using cuDNN kernels.
 */
template<typename DType>
class CuDNNNormConvolutionOp {
 public:
  CuDNNNormConvolutionOp()
#if CUDNN_VERSION >= 7600
      : fwd_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS),
        bwd_wgrad_op_(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD),
        fnlz_train_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING),
        fnlz_inference_op_(CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE)
#endif
  {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&equiv_scale_bias_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_stats_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_stats_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc_));
    parallelize_backward_kernels_ = Context::GetGPUStreamsPerWorker() >= 2;
  }

  ~CuDNNNormConvolutionOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(equiv_scale_bias_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_stats_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_stats_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc_));
    if (init_feature_vector_constants_) {
      init_feature_vector_constants_ = false;
      // We're probably at program exit, bypass the conventional approach of
      //     mxnet::Storage::Get()->DirectFree(zeros_feature_vector_hdl_);
      CUDA_CALL(cudaFree(zeros_feature_vector_hdl_.dptr));
      CUDA_CALL(cudaFree(ones_feature_vector_hdl_.dptr));
    }
  }

  void Init(const NormConvolutionParam& param,
            bool output_stats,
            const std::vector<TShape>& in_shapes,
            const TShape& out_shape,
            const OpContext& ctx) {
#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    using namespace mshadow;
    this->param_ = param;
    this->fwd_op_plan_output_stats_ = output_stats;
    InitBufferForParam();
    auto cudnn_fwd_compute_type = convertToCuDNNDataType(mshadow::kFloat32);
    dtype_ = DataType<DType>::kCudnnFlag;
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? kFloat32 : DataType<DType>::kFlag;

    auto effective_layout = param_.layout.value();
    switch (effective_layout) {
      // 1D normalizedConvolutions will be executed as 2D normalizedConvolutions with a height of 1.
      case mshadow::kNCW: effective_layout = mshadow::kNCHW; break;
      case mshadow::kNWC: effective_layout = mshadow::kNHWC; break;
      case mshadow::kCWN: effective_layout = mshadow::kCHWN; break;
      default: break;
    }

    MSHADOW_LAYOUT_SWITCH(effective_layout, Layout, {
      format_ = LayoutType<Layout>::kCudnnFlag;
    });

    // Double check to make sure this class supports the operation
    if (!Supports(param, in_shapes[norm_conv::kData], ctx.run_ctx.ctx.dev_id))
      LOG(FATAL) << "Unexpected unsupported use of NormConvolution op.";

    InitDescriptors(in_shapes, out_shape, cudnn_fwd_compute_type, output_stats, ctx);

    // Have cuDNN make a 'plan' for the fused op, returning the temp workspace size required.
    GetTempSize(ctx);

    // Create an equivalent BatchNormParam for the held instance of the NhwcBatchNormOp
    // Not needed for Backward
    bn_param_.eps = 0.0;
    // Not needed for Backward since running mean/var are updated by forward kernel.
    bn_param_.momentum = 0.f;
    // Finalize kernel can respond to fix_gamma = true
    bn_param_.fix_gamma = false;
    // use_global_stats will only be true for inference-only graphs where backward is not needed
    bn_param_.use_global_stats = false;
    // Should have no effect on NHWCBatchNorm::Backward()
    bn_param_.output_mean_var = true;
    // NormConvolution only supported for NHWC layouts
    CHECK_EQ(effective_layout, mshadow::kNHWC);
    bn_param_.axis = 3;
    // Only cudnn NormConvolution is implemented
    bn_param_.cudnn_off = false;
    // Copy act_type value from NormalizeConvolutionParam -> BatchNormParam
    if (param_.act_type.has_value())
      bn_param_.act_type = param_.act_type;
    bn_param_.bn_group = 1;
    bn_param_.xbuf_ptr = 0U;

    if (!param_.no_norm) {
      int in_features = static_cast<int>(Features(in_shapes[norm_conv::kData]));
      FinalizeInit(param_, Shape1(in_features), ctx);
    }

#endif  // CUDNN_VERSION >= 7600
  }


  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    size_t expected_inputs = norm_conv::NumInputs(param_.no_norm);
    size_t expected_outputs = norm_conv::NumOutputs(param_.no_norm);
    CHECK_EQ(in_data.size(), expected_inputs);
    CHECK_EQ(out_data.size(), expected_outputs);
    CHECK_EQ(req.size(), expected_outputs);
    // Finalize checks
    if (!param_.no_norm) {
      CHECK(req[norm_conv::kEquivScale] == kWriteTo || req[norm_conv::kEquivScale] == kNullOp);
      CHECK(req[norm_conv::kEquivBias] == kWriteTo || req[norm_conv::kEquivBias] == kNullOp);
    }
    // All inputs (except for the data and weight) should have
    // a shape equal to the one used to init the op
    for (size_t i = 0; i != in_data.size(); ++i) {
      if (i == static_cast<size_t>(norm_conv::kData) ||
          i == static_cast<size_t>(norm_conv::WeightIdx(param_.no_norm)))
        continue;
      CHECK_EQ(in_data[i].shape_, init_shape_);
    }
    // All outputs (except for the first 3) should have a shape equal to the one used to init the op
    for (size_t i = 3; i != out_data.size(); ++i) {
      CHECK_EQ(out_data[i].shape_, init_shape_);
    }

    if (!param_.no_norm)
      FinalizeForward(ctx, in_data, req, out_data);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, fwd_workspace_byte_);
    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    int weight_idx = norm_conv::WeightIdx(param_.no_norm);
    DType *data_ptr = GetNdPtr(in_data[norm_conv::kData], param_.kernel.ndim() + 2, s);
    DType *wmat_ptr = GetNdPtr(in_data[weight_idx], param_.kernel.ndim() + 2, s);

    int in_features = static_cast<int>(Features(in_data[norm_conv::kData].shape_));
    DType *equiv_scale_ptr = nullptr;
    DType *equiv_bias_ptr = nullptr;
    if (fprop_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) {
      if (param_.no_norm) {
        equiv_scale_ptr = static_cast<DType *>(ones_feature_vector_hdl_.dptr);
        equiv_bias_ptr = static_cast<DType *>(zeros_feature_vector_hdl_.dptr);
      } else {
        equiv_scale_ptr = out_data[norm_conv::kEquivScale].get_with_shape<gpu, 1, DType>(
                              Shape1(in_features), s).dptr_;
        equiv_bias_ptr = out_data[norm_conv::kEquivBias].get_with_shape<gpu, 1, DType>(
                             Shape1(in_features), s).dptr_;
      }
    }

    DType *out_ptr = GetNdPtr(out_data[norm_conv::kOut], param_.kernel.ndim() + 2, s);
    // Make sure the op's present need for output_stats corresponds to the assumed need at the time
    // the 'plan' was made.
    bool needed_output_stats = CuDNNNormConvolutionOp::OutputStats(ctx, req);
    CHECK_EQ(needed_output_stats, fwd_op_plan_output_stats_) <<
      "Improper instance lookup for CuDNNNormConvolutionOp: improper 'output_stats' bool.";

    int out_features = static_cast<int>(Features(out_data[norm_conv::kOut].shape_));
    // No implementations of this op exist with DType = double, so output stats pointers
    // will always be float.
    float *sum_ptr = nullptr;
    float *sum_of_squares_ptr = nullptr;
    if (fwd_op_plan_output_stats_) {
      sum_ptr = out_data[norm_conv::kOutSum].get_with_shape<gpu, 1, float>(
          Shape1(out_features), s).dptr_;
      sum_of_squares_ptr = out_data[norm_conv::kOutSumOfSquares].get_with_shape<gpu, 1, float>(
          Shape1(out_features), s).dptr_;
    }

    CHECK_EQ(req[norm_conv::kOut], kWriteTo) <<
      "In norm-conv output, expecting simple write of output, not add-to or inplace write.";

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    // This operator does not support output blending as specified by alpha or beta.
    // Set data input pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, data_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WDATA, wmat_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    // Set workspace input pointer in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace.dptr_);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &workspace_size);
    // Set data output pointers in op instance
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YDATA, out_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum_ptr);
    fwd_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_of_squares_ptr);
    // Launch forward operation
    fwd_op_.Execute(s->dnn_handle_);
#endif  // CUDNN_VERSION < 7600
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<TBlob> &fwd_in_data,
                const std::vector<TBlob> &fwd_out_data,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    size_t expected_inputs = norm_conv::NumInputs(param_.no_norm);
    CHECK_EQ(fwd_in_data.size(), expected_inputs);
    // We expect to see an in_grad tensor for all inputs,
    // except the moving_mean and moving_var 'aux inputs'.
    CHECK_EQ(in_grad.size(), norm_conv::NumNonAuxInputs(param_.no_norm));
    Stream<gpu> *s = ctx.get_stream<gpu>();
    // RAII object to handle syncing of the underlying auxiliary stream with the primary stream
    SyncedGPUAuxStream s_wgrad = ctx.get_gpu_aux_stream();
    size_t dgrad_kernels_workspace_offset_byte =
        parallelize_backward_kernels_ ? bwd_wgrad_workspace_byte_ : 0;
    // The temp space bytes requested to cover the kernels called directly by this routine (the
    // wgrad and conv-dgrag).  The nhwc_bn_op.Backward() will also request a workspace to cover
    // the bn-dgrad (+ potentially the wgrad if parallelize_backward_kernels_ is true).
    size_t backward_workspace_byte =
        parallelize_backward_kernels_ ? bwd_wgrad_workspace_byte_ + bwd_dgrad_conv_workspace_byte_
                                      : std::max(bwd_wgrad_workspace_byte_,
                                                 bwd_dgrad_conv_workspace_byte_);
    Tensor<gpu, 1, DType> workspace = AllocateTempWorkspace(ctx, backward_workspace_byte);
    size_t workspace_size = TensorSizeBytes(workspace);

    // I/O's should have 2 more dims than the kernel dim
    int input_weight_idx = norm_conv::WeightIdx(param_.no_norm);
    int ingrad_weight_idx = norm_conv::BwdInGradIdx(input_weight_idx);
    // Ptr to the forward operation data input
    DType *data_ptr = GetNdPtr(fwd_in_data[norm_conv::kData], param_.kernel.ndim() + 2, s);
    // Ptr to the incoming gradient of the forward operation data output 'Y' (an input here)
    DType *y_grad_ptr = GetNdPtr(out_grad[norm_conv::kOut], param_.kernel.ndim() + 2, s);
    // Ptr to the outgoing gradient of the forward operation weight input (an output here)
    DType *wgt_grad_ptr = GetNdPtr(in_grad[ingrad_weight_idx], param_.kernel.ndim() + 2, s);

    int in_features = static_cast<int>(Features(fwd_in_data[norm_conv::kData].shape_));
    DType *equiv_scale_ptr = nullptr;
    DType *equiv_bias_ptr = nullptr;
    if (wgrad_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) {
      if (param_.no_norm) {
        equiv_scale_ptr = static_cast<DType *>(ones_feature_vector_hdl_.dptr);
        equiv_bias_ptr = static_cast<DType *>(zeros_feature_vector_hdl_.dptr);
      } else {
        equiv_scale_ptr = fwd_out_data[norm_conv::kBwdEquivScale].get_with_shape<gpu, 1, DType>(
                              Shape1(in_features), s).dptr_;
        equiv_bias_ptr = fwd_out_data[norm_conv::kBwdEquivBias].get_with_shape<gpu, 1, DType>(
                             Shape1(in_features), s).dptr_;
      }
    }
#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    // WGRAD

    // This operator does not support output blending as specified by alpha or beta.
    // Set data input pointers in op instance
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_XDATA, data_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DYDATA, y_grad_ptr);
    // Here we supply equiv_scale and equiv_bias ptrs, though perhaps statically-initted ones
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale_ptr);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias_ptr);
    // Set workspace input pointer in op instance
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, workspace.dptr_);
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
                                           &workspace_size);
    // Set data output pointers in op instance
    bwd_wgrad_op_.SetOpVariantParamAttrPtr(CUDNN_PTR_DWDATA, wgt_grad_ptr);
    // Launch backward wgrad operation into the alternate stream (if enabled)
    bwd_wgrad_op_.Execute(s_wgrad.GetStream()->dnn_handle_);

    // DGRAD - convolution dgrad followed optionally by batchnorm dgrad

    if (req[conv::kData] != kNullOp) {
      // First: convolution dgrad
      typename DataType<DType>::ScaleType alpha = 1.0f;
      typename DataType<DType>::ScaleType beta = 0.0f;
      typename DataType<DType>::ScaleType beta_add = 1.0f;
      bool only_conv_dgrad_present = param_.no_norm && !param_.act_type.has_value();
      typename DataType<DType>::ScaleType conv_dgrad_beta =
        (only_conv_dgrad_present && (req[norm_conv::kData] == kAddTo)) ? beta_add : beta;
      if (!only_conv_dgrad_present && (req[norm_conv::kData] == kAddTo))
        LOG(FATAL) << "NormConvolution dgrad output summation not supported "
                      "when applying stats (needs extra conv dgrad buffer not yet allocated).";

      // Ptr to the forward operation weight input
      DType *wmat_ptr = GetNdPtr(fwd_in_data[input_weight_idx], param_.kernel.ndim() + 2, s);
      // Ptr to the outgoing gradient of the forward operation data input (an output here)
      DType *x_grad_ptr = GetNdPtr(in_grad[norm_conv::kData], param_.kernel.ndim() + 2, s);
      DType *dgrad_workspace_ptr =  workspace.dptr_
                                    + dgrad_kernels_workspace_offset_byte / sizeof(DType);
      size_t dgrad_workspace_byte = workspace_size - dgrad_kernels_workspace_offset_byte;
      // Launch conv dgrad into the primary stream, although with an offsetted workspace
      // pointer if dual-stream is enabled.
      CUDNN_CALL(cudnnConvolutionBackwardData(s->dnn_handle_,
          &alpha,
          filter_desc_,
          wmat_ptr,
          out_desc_,
          y_grad_ptr,
          conv_desc_,
          back_conv_dgrad_algo_,
          dgrad_workspace_ptr,
          dgrad_workspace_byte,
          &conv_dgrad_beta,
          in_desc_,
          x_grad_ptr));
      // Second (if needed): batchnorm dgrad
      if (!only_conv_dgrad_present) {
        // The following unusual case is not typically found (e.g. in Resnet).
        if (param_.no_norm)
          LOG(FATAL) << "Relu activation with no_norm not yet supported.";
        // Prepare inputs of NHWCBatchnorm::Backward()
        // Note that the 1st input is the same as the 1st output, i.e. the Batchnorm
        // is operating 'in place' on the gradient as output by the convolution dgrad.
        TBlob not_used;
        std::vector<TBlob> bn_bwd_inputs{in_grad[norm_conv::kData],
                                         fwd_out_data[norm_conv::kBwdSavedMean],
                                         fwd_out_data[norm_conv::kBwdSavedInvStdDev],
                                         fwd_in_data[norm_conv::kData],
                                         fwd_in_data[norm_conv::kGamma],
                                         fwd_in_data[norm_conv::kBeta],
                                         not_used,
                                         not_used};
        std::vector<OpReqType> bn_bwd_req{req[norm_conv::kData],
                                          req[norm_conv::BwdInGradIdx(norm_conv::kGamma)],
                                          req[norm_conv::BwdInGradIdx(norm_conv::kBeta)]};
        std::vector<TBlob> bn_bwd_outputs{in_grad[norm_conv::kData],
                                          in_grad[norm_conv::BwdInGradIdx(norm_conv::kGamma)],
                                          in_grad[norm_conv::BwdInGradIdx(norm_conv::kBeta)]};
        // This function will ask for a temp workspace and will get the same pointer
        // (as long as the workspace does not to be increased).  This all works fine because
        // at this point the wgrad, conv-dgad and this bn-dgrad are all using the same stream.

        // The Init call is made prior to each Backward(), a historical result of transitioning
        // from a symbolic to a gluon (imperative) op style.
        nhwc_bn_op.Init(bn_param_);
        // Launch batchnorm backward into the primary stream.  This will launch a kernel with
        // an offsetted workspace pointer if dual-stream is enabled.
        nhwc_bn_op.Backward(ctx, bn_bwd_inputs, bn_bwd_req, bn_bwd_outputs,
                            dgrad_kernels_workspace_offset_byte);
      }
    }
#endif  // CUDNN_VERSION < 7600
  }

  /*!
   * \brief Returns whether the norm convolution operation described by `param`
   * is supported.
   */
  template <typename SupportedConvParam>
  static bool Supports(const SupportedConvParam &param,
                       const TShape& in_data_shape,
                       int dev_id) {
    using namespace mshadow;
    static_assert(std::is_same<SupportedConvParam, ConvolutionParam>::value ||
                  std::is_same<SupportedConvParam, NormConvolutionParam>::value,
                  "Unsupported template specialization of NormConvolution::Supports()");
    // Need cuDNN version >= 7.6
    if (CUDNN_VERSION < 7600)
      return false;
    // Only Volta GPU arch (70) specifically supported.  Not earlier arches or Turing (75).
    if (SMArch(dev_id) != 70)
      return false;
    // Only kNHWC and kNWC format supported
    auto layout_val = param.layout.value();
    if (layout_val != kNWC && layout_val != kNHWC)
      return false;
    // Only 2D convolution supported
    if (param.kernel.ndim() != 2)
      return false;
    // Only 1x1 with no stride, or strided 3x3 supported
    if (!(param.kernel == TShape{3, 3}) &&
        !(param.kernel == TShape{1, 1} && param.stride == TShape{1, 1}))
      return false;
    // No dilation supported
    if (param.dilate != TShape{1, 1})
      return false;
    // No grouped convolution supported
    if (param.num_group != 1)
      return false;
    // Must have a multiple of 32 input features 'c' (assumes N..C layout).
    if (in_data_shape[in_data_shape.ndim()-1] % 32 != 0)
      return false;
    // Must have a multiple of 32 output features (== number of filters 'k')
    if (param.num_filter % 32 != 0)
      return false;
    // Op parameters are supported, assuming datatype is float16
    return DataType<DType>::kFlag == kFloat16;
  }

  // Does the operator need to emit valid 'sum' and 'sum_of_squares' tensors?
  static bool OutputStats(const OpContext &ctx, const std::vector<OpReqType> &req) {
    // In processing validation samples on the training graph, req[] values
    // are equal to kWriteOp, but the outputs are not needed.
    // ctx.is_train == false identifies this case.
    return ctx.is_train && (NeedsOutput(norm_conv::kOutSum, req) ||
                            NeedsOutput(norm_conv::kOutSumOfSquares, req));
  }

 private:
  static bool NeedsOutput(size_t output_index, const std::vector<OpReqType> &req) {
    return (req.size() > output_index) && (req[output_index] != kNullOp);
  }

  // Converts a TBlob to a 1D dptr of the specifed size, checking for that it's contiguous.
  DType *Get1dPtr(const TBlob &tb, int size, mshadow::Stream<gpu> *s) {
    mshadow::Tensor<gpu, 1, DType> data = tb.get<gpu, 1, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    return data.dptr_;
  }

/*!
 * \brief Translate an mxnet datatype to the corresponding cudnnDataType_t.
 */
  cudnnDataType_t convertToCuDNNDataType(int dtype) {
    cudnnDataType_t converted = CUDNN_DATA_FLOAT;
    // The following will always assign to `converted` or throw an exception.
    MSHADOW_REAL_TYPE_SWITCH(dtype, mxDType, {
      converted = mshadow::DataType<mxDType>::kCudnnFlag;
    })
    return converted;
  }

  void FinalizeInit(const NormConvolutionParam &param, const TShape shape,
            const OpContext &ctx) {
    CHECK_GE(param.eps, CUDNN_BN_MIN_EPSILON)
     << "CuDNN requires eps to be no less than " << CUDNN_BN_MIN_EPSILON;
    FinalizeInitDescriptors(shape);

#if CUDNN_VERSION >= 7600
    // Set up the 'Const Param Pack' for the BNForwardFinalizeStatisticsTraining op
    // Describe pointer alignments
    fnlz_train_op_.SetOpConstParamAttr({CUDNN_PARAM_YSUM_PLACEHOLDER,
                                   CUDNN_PARAM_YSQSUM_PLACEHOLDER,
                                   CUDNN_PARAM_BN_SCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
                                   CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER,
                                   CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER,
                                   CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
                                   CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
                                   CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                   CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER}, CUDNN_PTR_ELEM_ALIGNED);
    // Set the I/O descriptors
    // sum and sum_squares input descriptor (typically fp32). Also
    // scale, bias, running_mean and running_var input descriptors, as well as the
    // saved_mean and saved_inv_std output descriptor (typically fp32)
    fnlz_train_op_.SetOpConstParamDesc({CUDNN_PARAM_YSTATS_DESC,
                                   CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC}, in_stats_desc_);
    // equiv_scale and equiv_bias output descriptor (typically fp16)
    fnlz_train_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC, equiv_scale_bias_desc_);

    // Set up the 'Const Param Pack' for the BNForwardFinalizeStatisticsInference op
    fnlz_inference_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_SCALE_PLACEHOLDER,
                                       CUDNN_PARAM_BN_BIAS_PLACEHOLDER,
                                       CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
                                       CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
                                       CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                       CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER}, CUDNN_PTR_ELEM_ALIGNED);
    // Set the I/O descriptors
    // scale, bias, running_mean and running_var input descriptors, as well as the
    // saved_mean and saved_inv_std output descriptor (typically fp32)
    fnlz_inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, in_stats_desc_);
    // equiv_scale and equiv_bias output descriptor (typically fp16)
    fnlz_inference_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC, equiv_scale_bias_desc_);

    // Perform some actions identically on both train and inference ops.
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    for (auto op : {&fnlz_train_op_, &fnlz_inference_op_}) {
      // Set the mode parameter in the ops, can't be CUDNN_BATCHNORM_PER_ACTIVATION.
      op->SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
      // Check workspace size, also creates 'plan'.
      size_t workspace_size_bytes = op->GetWorkspaceSizeInBytes(s->dnn_handle_);
      CHECK_EQ(workspace_size_bytes, 0U)
        << "Unexpected non-zero workspace size for CuDNNBNStatsFinalize op.";
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, static_cast<void *>(nullptr));
      op->SetOpVariantParamAttrPtr(CUDNN_PTR_WORKSPACE, &workspace_size_bytes);
    }
#endif
  }

  void FinalizeForward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;

    int in_features = in_data[norm_conv::kInSum].shape_[0];
    auto in_feature_shape = Shape1(in_features);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    DType *equiv_scale = Get1dPtr(out_data[norm_conv::kEquivScale], in_features, s);
    DType *equiv_bias = Get1dPtr(out_data[norm_conv::kEquivBias], in_features, s);

#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> sum = in_data[norm_conv::kInSum]
          .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
      Tensor<gpu, 1, DTypeParam> sum_squares = in_data[norm_conv::kInSumOfSquares]
          .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
      Tensor<gpu, 1, DTypeParam> gamma = in_data[norm_conv::kGamma]
          .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
      Tensor<gpu, 1, DTypeParam> beta = in_data[norm_conv::kBeta]
          .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
      Tensor<gpu, 1, DTypeParam> moving_mean = in_data[norm_conv::kMovingMean]
          .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
      Tensor<gpu, 1, DTypeParam> moving_inv_var = in_data[norm_conv::kMovingVar]
          .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);

      if (param_.fix_gamma) gamma = 1.f;

      auto &op = ctx.is_train ? fnlz_train_op_ : fnlz_inference_op_;

      // The prep needed for the train_op_ is a superset of that needed for the inference_op_.
      // Start here with the common prep needed for the inference_op_:
      // Set data pointers in the 'variant param pack'
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SCALE, gamma.dptr_);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_BIAS, beta.dptr_);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_MEAN, moving_mean.dptr_);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_RUNNING_VAR, moving_inv_var.dptr_);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQSCALE, equiv_scale);
      op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_EQBIAS, equiv_bias);
      // Set some additional light-weight parameters in the 'variant param pack'
      op.SetOpVariantParamAttrPtr<double>(CUDNN_SCALAR_DOUBLE_BN_EPSILON, &param_.eps);

      // Now add additional prep needed only for train_op_:
      if (ctx.is_train) {
        Tensor<gpu, 1, DTypeParam> save_mean = out_data[norm_conv::kSavedMean]
            .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
        Tensor<gpu, 1, DTypeParam> save_inv_var = out_data[norm_conv::kSavedInvStdDev]
            .get_with_shape<gpu, 1, DTypeParam>(in_feature_shape, s);
        // Set data pointers in the 'variant param pack'
        op.SetOpVariantParamAttrPtr(CUDNN_PTR_YSUM, sum.dptr_);
        op.SetOpVariantParamAttrPtr(CUDNN_PTR_YSQSUM, sum_squares.dptr_);
        op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_MEAN, save_mean.dptr_);
        op.SetOpVariantParamAttrPtr(CUDNN_PTR_BN_SAVED_INVSTD, save_inv_var.dptr_);
        // Set some additional light-weight parameters in the 'variant param pack'
        int64_t elem_count = in_data[norm_conv::kData].shape_.Size() / in_features;
        double avg_factor = 1.0 - param_.momentum;
        op.SetOpVariantParamAttrPtr(CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT, &elem_count);
        op.SetOpVariantParamAttrPtr(CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, &avg_factor);
      }
      // Finally, launch op
      op.Execute(s->dnn_handle_);
    })
#endif  // CUDNN_VERSION >= 7600
  }

  void FinalizeInitDescriptors(const TShape &shape) {
    using namespace mshadow;
    init_shape_ = shape;
    CHECK_EQ(init_shape_.ndim(), 1) << "Expecting 1D 'sum' input.";
    int c = init_shape_[0];
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      CUDNN_CALL(cudnnSetTensor4dDescriptor(in_stats_desc_,
                                            format,
                                            DataType<DTypeParam>::kCudnnFlag,
                                            1, c, 1, 1));
    })
  }

  void InitDescriptors(const std::vector<TShape>& in_shapes,
                       const TShape& out_shape,
                       cudnnDataType_t cudnn_fwd_compute_type,
                       bool output_stats,
                       const OpContext& ctx) {
    using namespace mshadow;
    size_t expected_inputs = norm_conv::NumInputs(param_.no_norm);
    CHECK_EQ(in_shapes.size(), expected_inputs);

    TShape dshape = in_shapes[norm_conv::kData];
    int weight_idx = norm_conv::WeightIdx(param_.no_norm);
    TShape wshape = in_shapes[weight_idx];
    TShape oshape = out_shape;
    TShape dstride, ostride;
#if CUDNN_VERSION < 7600
    LOG(FATAL) << "cuDNN version 7.6 or later is required.";
#else
    bool no_activation = !param_.act_type.has_value();
    // We'll need normconv fprop if we're doing a normalization, activation, or outputting stats.
    bool fused_normconv_fprop_needed = !param_.no_norm || !no_activation || output_stats;
    // We only supply a null pointer if we think cuDNN can then fall back to the conventional fprop.
    fprop_eq_scale_bias_ptr_type_ = fused_normconv_fprop_needed ? CUDNN_PTR_16B_ALIGNED
                                                                : CUDNN_PTR_NULL;

    // We'll need normconv wgrad if we're doing a normalization or activation.
    bool fused_normconv_wgrad_needed = !param_.no_norm || !no_activation;
    // We only supply a null pointer if we think cuDNN can then fall back to the conventional wgrad.
    wgrad_eq_scale_bias_ptr_type_ = fused_normconv_wgrad_needed ? CUDNN_PTR_16B_ALIGNED
                                                                : CUDNN_PTR_NULL;
    auto stats_ptr_type = output_stats ? CUDNN_PTR_16B_ALIGNED : CUDNN_PTR_NULL;

    // Describe i/o tensor pointer alignment for forward fused op
    fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_XDATA_PLACEHOLDER,
                                 CUDNN_PARAM_WDATA_PLACEHOLDER,
                                 CUDNN_PARAM_YDATA_PLACEHOLDER}, CUDNN_PTR_16B_ALIGNED);
    fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                 CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                     fprop_eq_scale_bias_ptr_type_);
    fwd_op_.SetOpConstParamAttr({CUDNN_PARAM_YSUM_PLACEHOLDER,
                                 CUDNN_PARAM_YSQSUM_PLACEHOLDER}, stats_ptr_type);

    // Describe i/o tensor pointer alignment for backward wgrad fused op
    bwd_wgrad_op_.SetOpConstParamAttr({CUDNN_PARAM_DYDATA_PLACEHOLDER,
                                       CUDNN_PARAM_XDATA_PLACEHOLDER,
                                       CUDNN_PARAM_DWDATA_PLACEHOLDER}, CUDNN_PTR_16B_ALIGNED);
    bwd_wgrad_op_.SetOpConstParamAttr({CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                                       CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER},
                                           wgrad_eq_scale_bias_ptr_type_);

    if (param_.kernel.ndim() == 1 || param_.kernel.ndim() == 2) {
      // 1d or 2d conv
      auto pad = param_.kernel.ndim() == 2 ? param_.pad : TShape({0, param_.pad[0]});
      auto stride = param_.kernel.ndim() == 2 ? param_.stride : TShape({1, param_.stride[0]});
      auto dilate = param_.kernel.ndim() == 2 ? param_.dilate : TShape({1, param_.dilate[0]});
      CUDNN_CALL(cudnnSetConvolution2dDescriptor(conv_desc_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CROSS_CORRELATION,
                                               cudnn_fwd_compute_type));
      if (param_.kernel.ndim() == 2) {
        wshape = ConvertLayout(wshape.get<4>(), param_.layout.value(), kNCHW);
        dstride = ConvertLayout(Strides<4>(dshape), param_.layout.value(), kNCHW);
        dshape = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW);
        ostride = ConvertLayout(Strides<4>(oshape), param_.layout.value(), kNCHW);
        oshape = ConvertLayout(oshape.get<4>(), param_.layout.value(), kNCHW);
      } else {
        wshape = ConvertLayout(wshape.get<3>(), param_.layout.value(), kNCW);
        wshape = TShape({wshape[0], wshape[1], 1, wshape[2]});
        dstride = ConvertLayout(Strides<3>(dshape), param_.layout.value(), kNCW);
        dstride = TShape({dstride[0], dstride[1], dstride[1], dstride[2]});
        dshape = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW);
        dshape = TShape({dshape[0], dshape[1], 1, dshape[2]});
        ostride = ConvertLayout(Strides<3>(oshape), param_.layout.value(), kNCW);
        ostride = TShape({ostride[0], ostride[1], ostride[1], ostride[2]});
        oshape = ConvertLayout(oshape.get<3>(), param_.layout.value(), kNCW);
        oshape = TShape({oshape[0], oshape[1], 1, oshape[2]});
      }
      CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                            dtype_,
                                            format_,
                                            wshape[0],
                                            wshape[1],
                                            wshape[2],
                                            wshape[3]));

    } else if (param_.kernel.ndim() == 3) {
      LOG(FATAL) << "3D NormConvolution not supported.";
    }
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
    if (GetEnvAllowTensorCore() && GetEnvAllowTensorCoreConversion() &&
        (DataType<DType>::kFlag != kFloat16))
      math_type = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc_, param_.num_group));

    std::vector<int> dshape_buffer(dshape.ndim());
    nnvm::ShapeTypeCast(dshape.begin(), dshape.end(), dshape_buffer.data());
    std::vector<int> dstride_buffer(dstride.ndim());
    nnvm::ShapeTypeCast(dstride.begin(), dstride.end(), dstride_buffer.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                          dtype_,
                                          static_cast<int>(dshape.ndim()),
                                          dshape_buffer.data(),
                                          dstride_buffer.data()));

    std::vector<int> oshape_buffer(oshape.ndim());
    nnvm::ShapeTypeCast(oshape.begin(), oshape.end(), oshape_buffer.data());
    std::vector<int> ostride_buffer(ostride.ndim());
    nnvm::ShapeTypeCast(ostride.begin(), ostride.end(), ostride_buffer.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
                                          static_cast<int>(oshape.ndim()),
                                          oshape_buffer.data(),
                                          ostride_buffer.data()));

    // Always set scale/bias descriptors
    int in_features = static_cast<int>(Features(in_shapes[norm_conv::kData]));
    TShape equiv_scale_bias_shape = TShape({in_features});
    std::vector<int> equiv_scale_shape = {1, static_cast<int>(in_features), 1, 1};
    std::vector<int> equiv_scale_stride = {static_cast<int>(in_features), 1, 1, 1};
    CUDNN_CALL(cudnnSetTensorNdDescriptor(equiv_scale_bias_desc_,
                                        dtype_,
                                        static_cast<int>(equiv_scale_shape.size()),
                                        &equiv_scale_shape[0],
                                        &equiv_scale_stride[0]));
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC, equiv_scale_bias_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_BN_EQSCALEBIAS_DESC, equiv_scale_bias_desc_);

    if (!param_.no_norm) {
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kInSum]) <<
        "Expecting equal equivalent-scale and sum input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kInSumOfSquares]) <<
        "Expecting equal equivalent-scale and sum_squares input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kMovingMean]) <<
        "Expecting equal equivalent-scale and saved-mean input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kMovingVar]) <<
        "Expecting equal equivalent-scale and saved-inv-stddev input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kGamma]) <<
        "Expecting equal equivalent-scale and gamma input tensor shapes.";
      CHECK_EQ(equiv_scale_bias_shape, in_shapes[norm_conv::kBeta]) <<
        "Expecting equal equivalent-scale and beta input tensor shapes.";
    }

    if (output_stats) {
    // Replace with checks at every Forward()?
//      TShape sum_shape = out_shape[norm_conv::kOutSum];
//      TShape sum_of_squares_shape = out_shape[norm_conv::kOutSumOfSquares];
//      CHECK_EQ(sum_shape, sum_of_squares_shape) <<
//        "Expecting equal sum and sum_of_squares output tensor shapes.";
      int output_features = static_cast<int>(Features(out_shape));
      std::vector<int> stats_shape = {1, output_features, 1, 1};
      std::vector<int> stats_stride = {output_features, 1, 1, 1};
      // Stats are output in the same precision as the forward compute (i.e. float32)
      CUDNN_CALL(cudnnSetTensorNdDescriptor(out_stats_desc_,
                                          cudnn_fwd_compute_type,
                                          static_cast<int>(stats_shape.size()),
                                          &stats_shape[0],
                                          &stats_stride[0]));
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YSTATS_DESC, out_stats_desc_);
    }

    // Here's where the standard convolution does a 'SelectAlgo', which may run cudnnFind()
    // Not available yet for the NormConvolution operation.
    // If we're allowing Tensor Core variants of the algos to be considered in

    // Copied temporarily from 'SelectAlgo': probably not needed

    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

    // Set activation descriptor, default is no activation
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_IDENTITY;
    if (param_.act_type.has_value()) {
      CHECK_EQ(param_.act_type.value(), activation::kReLU) <<
        "Only relu activation supported in normalized convolution.";
      mode = CUDNN_ACTIVATION_RELU;
    }
    auto nan_prop = CUDNN_NOT_PROPAGATE_NAN;
    double dummy_clip = 0.0;
    CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc_, mode, nan_prop, dummy_clip));
    // Currently, the only way to turn off activation is to not set the descriptor
    if (mode != CUDNN_ACTIVATION_IDENTITY) {
      fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC, activation_desc_);
      bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_ACTIVATION_DESC, activation_desc_);
    }

    // Set desc pointers
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_XDESC, in_desc_);
    // FusedOp does not accept CUDNN_BATCHNORM_PER_ACTIVATION
    fwd_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);
    bwd_wgrad_op_.SetOpConstParamAttr(CUDNN_PARAM_BN_MODE, CUDNN_BATCHNORM_SPATIAL);

    // The Cudnn Convolution op provides parameters for controlling math precision
    // separately for forward and backward, and so there are separate forward and backward conv
    // descriptors.  However, NormConvolution does not have these extra parameters, so the
    // same descriptor can be used for both.
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_CONV_DESC, conv_desc_);
    // W desc for forward == dW desc for backward wgrad
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_WDESC, filter_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_DWDESC, filter_desc_);
    // Y desc for forward == dY desc for backward wgrad
    fwd_op_.SetOpConstParamDesc(CUDNN_PARAM_YDESC, out_desc_);
    bwd_wgrad_op_.SetOpConstParamDesc(CUDNN_PARAM_DYDESC, out_desc_);
#endif  // CUDNN_VERSION < 7600

    // Set up 0.f and 1.f constant vectors if we're running in no-apply mode
    if ((fprop_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL ||
         wgrad_eq_scale_bias_ptr_type_ != CUDNN_PTR_NULL) && param_.no_norm
                                                          && !init_feature_vector_constants_) {
      size_t equiv_scale_bytes = in_features * sizeof(DType);
      Stream<gpu> *s = ctx.get_stream<gpu>();
      zeros_feature_vector_hdl_ = mxnet::Storage::Get()->Alloc(equiv_scale_bytes, Context::GPU());
      // Zero the read-only zeros_feature_vector_hdl_ area once.
      CUDA_CALL(cudaMemsetAsync(zeros_feature_vector_hdl_.dptr, 0,
                                equiv_scale_bytes, mshadow::Stream<gpu>::GetStream(s)));
      ones_feature_vector_hdl_ = mxnet::Storage::Get()->Alloc(equiv_scale_bytes, Context::GPU());
      // Setting this up as 1's is a little tricky.  Not sure if the cuMemsetD32Async would
      // have endian issues.
      TBlob ones_tblob(ones_feature_vector_hdl_.dptr, equiv_scale_bias_shape, gpu::kDevMask,
                       DataType<DType>::kFlag, ctx.run_ctx.ctx.dev_id);
      auto ones_tensor = ones_tblob.get_with_shape<gpu, 1, DType>(Shape1(in_features), s);
      // Now init the ones tensor
      ones_tensor = 1.f;
      init_feature_vector_constants_ = true;
    }
  }

  void GetTempSize(const OpContext& ctx) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    // Make op plan for forward op and set forward workspace size
    fwd_workspace_byte_ = fwd_op_.GetWorkspaceSizeInBytes(s->dnn_handle_);
    // Make op plan for backward wgrad op and set backward wgrad workspace size
    bwd_wgrad_workspace_byte_ = bwd_wgrad_op_.GetWorkspaceSizeInBytes(s->dnn_handle_);
    // Get workspace for backward dgrad- convolution requirement
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(s->dnn_handle_,
                                                            filter_desc_,
                                                            out_desc_,
                                                            conv_desc_,
                                                            in_desc_,
                                                            back_conv_dgrad_algo_,
                                                            &bwd_dgrad_conv_workspace_byte_));
    // cudaMalloc returns addresses that are aligned for large accesses (e.g. to 512 bytes).
    // Since we may make one allocation and divide it into two parts when we parallelize
    // the dgrad and wgrad kernels, we round the size of the wgrad tempspace up to this
    // alignment size so the temp space dptrs for the dgrad kernels will respect this alignment
    // when stacked on top of the wgrad temp area.
    const size_t dptr_alignment = 512;
    bwd_wgrad_workspace_byte_ = RoundToMultiple(bwd_wgrad_workspace_byte_, dptr_alignment);
  }

  int *CastTShapeToIntPtr(const TShape& s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    nnvm::ShapeTypeCast(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, Stream<gpu> *s) {
    DType *data_ptr = NULL;
    if (dim == 3) {
      Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

  // Converts a TShape to a Shape<> of strides.
  // e.g. {shape[0], shape[1], shape[2]} -> {shape[1]*shape[2], shape[2], 1}
  template <int dim>
  inline Shape<dim> Strides(const TShape &s) {
    uint32_t ndim = s.ndim();
    TShape strides(ndim, -1);
    for (uint32_t i = 0; i != ndim; ++i)
      strides[i] = s.ProdShape(i+1, ndim);
    return strides.get<dim>();
  }

  void InitBufferForParam() {
    CastTShapeToIntPtr(param_.stride, &param_stride_);
    CastTShapeToIntPtr(param_.dilate, &param_dilate_);
    CastTShapeToIntPtr(param_.pad, &param_pad_);
  }

  // Round a value 'x' up to the next multiple of 'multiple'
  size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, DType> AllocateTempWorkspace(const OpContext &ctx, size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words =
      std::max<size_t>(1, RoundToMultiple(size_bytes, sizeof(DType)) / sizeof(DType));
    return ctx.requested[norm_conv::kTempSpace].get_space_typed<gpu, 1, DType>(
        mshadow::Shape1(size_words), s);
  }

  // Returns the size in bytes of the 1D Tensor of words.
  size_t TensorSizeBytes(const mshadow::Tensor<gpu, 1, DType> &tensor) {
    return tensor.MSize() * sizeof(DType);
  }

  // Given a tensor shape of this operation, return the number of features 'c'
  int64_t Features(const TShape &dshape) {
    int c = 0;
    switch (dshape.ndim()) {
      case 3: c = ConvertLayout(dshape.get<3>(), param_.layout.value(), kNCW)[1]; break;
      case 4: c = ConvertLayout(dshape.get<4>(), param_.layout.value(), kNCHW)[1]; break;
      case 5: c = ConvertLayout(dshape.get<5>(), param_.layout.value(), kNCDHW)[1]; break;
      default:
        LOG(FATAL) << "Unexpected convolution data dimension " << dshape.ndim();
    }
    return c;
  }

  std::vector<int> param_stride_;
  std::vector<int> param_dilate_;
  std::vector<int> param_pad_;

  // Temp workspace size in bytes needed for Forward() operation.
  size_t fwd_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() wgrad operation.
  size_t bwd_wgrad_workspace_byte_;
  // Temp workspace size in bytes needed for Backward() dgrad operation (conv portion).
  size_t bwd_dgrad_conv_workspace_byte_;
  // The hardwired backward dgrad convolution algo
  cudnnConvolutionBwdDataAlgo_t back_conv_dgrad_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  // Should dgrad and wgrad be launched into separate streams
  bool parallelize_backward_kernels_;

  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t equiv_scale_bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t in_stats_desc_;
  cudnnTensorDescriptor_t out_stats_desc_;
  // Convolution descriptor for forward and backward operation (same math type used in both)
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorFormat_t format_;
  NormConvolutionParam param_;
  // The assumption of the fwd_op plan as to whether sum and sum_of_squares outputs are populated.
  bool fwd_op_plan_output_stats_;
  // A cached copy of the fwd_op plan ptr placeholder for equiv_stats and equiv_bias.
  cudnnFusedOpsPointerPlaceHolder_t fprop_eq_scale_bias_ptr_type_;
  // A cached copy of the bwd_op plan ptr placeholder for equiv_stats and equiv_bias.
  cudnnFusedOpsPointerPlaceHolder_t wgrad_eq_scale_bias_ptr_type_;

  // An instance of the equivalent Batchnorm operation, suitable for calling Backward() on.
  NhwcBatchNormOp<DType> nhwc_bn_op;
  // The BatchNormParam associated with the NHWCBatchNormOp instance
  BatchNormParam bn_param_;

  bool init_feature_vector_constants_ = false;
  mxnet::Storage::Handle zeros_feature_vector_hdl_;
  mxnet::Storage::Handle ones_feature_vector_hdl_;

  // Specifies activation parameters: relu
  cudnnActivationDescriptor_t activation_desc_;

  // data members to support finalize function
  int dtype_param_;
  // The shape used to init the in_stats_desc of the finalize op
  TShape init_shape_;

  #if CUDNN_VERSION >= 7600
  // New normalized convolution forward fused-op
  CuDNNCommonOp fwd_op_;
  // New normalized convolution backward wgrad fused-op
  CuDNNCommonOp bwd_wgrad_op_;
  // New 'fused op' for BN stats finalize forward (training mode)
  CuDNNCommonOp fnlz_train_op_;
  // New 'fused op' for BN stats finalize forward (inference mode)
  CuDNNCommonOp fnlz_inference_op_;
#endif
};
#endif  // __CUDACC__ && CUDNN
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_NORM_CONVOLUTION_INL_H_
