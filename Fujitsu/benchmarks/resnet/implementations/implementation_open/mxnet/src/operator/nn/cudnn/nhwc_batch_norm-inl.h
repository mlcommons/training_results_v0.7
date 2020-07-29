/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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
 * \file nhwc_batch_norm-inl.h
 * \brief
 * \author Dick Carter and Junyuan Xie
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_INL_H_
#include <mxnet/storage.h>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <cmath>
#include <algorithm>
#include "../batch_norm-inl.h"
#include "nhwc_batch_norm.h"
#include "../../../common/cuda_utils.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
namespace nhwcbatchnorm {
enum NhwcBatchNormOpInputs {kData, kGamma, kBeta};
enum NhwcBatchNormOpOutputs {kOut, kMean, kInvVar};
enum NhwcBatchNormOpAuxiliary {kMovingMean, kMovingInvVar};
enum NhwcBatchNormOpResource {kTempSpace};
}  // namespace nhwcbatchnorm

#if defined(__CUDACC__)
template<typename DType>
class NhwcBatchNormOp {
 public:
  typedef typename mshadow::DataType<DType>::ScaleType ScaleType;
  NhwcBatchNormOp() {
    using namespace mshadow;
    dtype_ = DataType<DType>::kCudnnFlag;
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? kFloat32 : DataType<DType>::kFlag;
  }

  void Init(const BatchNormParam &param) {
    if (param.act_type.has_value())
      CHECK_EQ(param.act_type.value(), activation::kReLU) <<
        "Only ReLU activation fusion supported.";
    if (!nhwc_layer_)
      nhwc_layer_.reset(new NhwcBatchNorm);

    this->param_ = param;

    workspace_total_bytes_ = 0;
  }

  ~NhwcBatchNormOp() {
    if (init_retired_ctas_) {
      init_retired_ctas_ = false;
      // We're probably at program exit, bypass the conventional approach of
      //     mxnet::Storage::Get()->DirectFree(retired_ctas_hdl_);
      CUDA_CALL(cudaFree(retired_ctas_hdl_.dptr));
    }
  }

  void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(aux_states.size(), 2U);
    if (ctx.is_train) {
      CHECK_EQ(out_data.size(), 3U);
      CHECK_EQ(req.size(), 3U);
    } else {
      CHECK_GE(out_data.size(), 1U);
      CHECK_GE(req.size(), 1U);
    }
    CHECK_EQ(req[nhwcbatchnorm::kOut], kWriteTo);
    CHECK_GE(in_data[nhwcbatchnorm::kData].ndim(), 2);
    CHECK_LE(in_data[nhwcbatchnorm::kData].ndim(), 5);

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Init(in_data[nhwcbatchnorm::kData], s, ctx.is_train);
    int features = in_data[nhwcbatchnorm::kData].shape_[param_.axis];

    DType *nhwc_X = GetNdPtr(in_data[nhwcbatchnorm::kData], dim_, s);
    DType *nhwc_dX = nullptr;
    DType *nhwc_Y = GetNdPtr(out_data[nhwcbatchnorm::kOut], dim_, s);
    DType *nhwc_dY = nullptr;

    nhwc_layer_->setInputOutputPointers(nhwc_X, nhwc_dX, nhwc_Y, nhwc_dY);

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_data[nhwcbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> beta =
        in_data[nhwcbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> moving_mean =
        aux_states[nhwcbatchnorm::kMovingMean]
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> moving_inv_var =
        aux_states[nhwcbatchnorm::kMovingInvVar]
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      std::vector<void *> weights = {gamma.dptr_, beta.dptr_};
      std::vector<void *> dweights = {nullptr, nullptr};
      nhwc_layer_->setWeightPointers(weights, dweights);

      std::vector<void *> params = {moving_mean.dptr_, moving_inv_var.dptr_};
      nhwc_layer_->setParameterPointers(params);

      if (param_.fix_gamma) gamma = 1.f;

      // MXNet already has allocated Tensors for the minibatch mean and variance.
      // These are the first two sizes returned by NhwcBatchNorm::numWorkspaceBytes
      // The other sizes must be allocated explicitly as temp workspace tensors
      std::vector<void*> workspace;

      if (ctx.is_train) {
        Tensor<gpu, 1, DTypeParam> save_mean =
          out_data[nhwcbatchnorm::kMean].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        Tensor<gpu, 1, DTypeParam> save_inv_var =
          out_data[nhwcbatchnorm::kInvVar]
            .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        workspace.push_back(save_mean.dptr_);
        workspace.push_back(save_inv_var.dptr_);
        workspace.push_back(retired_ctas_hdl_.dptr);
        nhwc_layer_->setBNGroup(param_.bn_group, reinterpret_cast<void**>(param_.xbuf_ptr),
                                ctx.run_ctx.ctx.dev_id);
      } else {
        workspace.push_back(nullptr);
        workspace.push_back(nullptr);
        workspace.push_back(nullptr);
      }

      mshadow::Tensor<gpu, 1, ScaleType> temp_space =
          this->AllocateTempWorkspace(ctx, workspace_total_bytes_);

      for (auto offset : workspace_byte_offsets_) {
        void *dptr = reinterpret_cast<char *>(temp_space.dptr_) + offset;
        workspace.push_back(dptr);
      }

      nhwc_layer_->setWorkspacePointers(workspace, workspace_bytes_);

      bool fuse_relu = param_.act_type.has_value() &&
        (param_.act_type.value() == activation::kReLU);
      if (ctx.is_train) {
        nhwc_layer_->fwd(Stream<gpu>::GetStream(s), fuse_relu, ctx.run_ctx.ctx.dev_id);
      } else {
        nhwc_layer_->fwdInference(Stream<gpu>::GetStream(s), fuse_relu);
      }
    })
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &outputs,
                size_t inuse_tempspace_bytes = 0U) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 8U);
    CHECK_EQ(outputs.size(), 3U);

    // Rename the inputs and outputs.
    const TBlob &out_grad = inputs[0];
    const TBlob &out_mean = inputs[1];
    const TBlob &out_var = inputs[2];
    const TBlob &in_data = inputs[3];
    const TBlob &in_gamma = inputs[4];
    const TBlob &in_beta = inputs[5];   // not sure if this is right @TODO check
    const std::vector<TBlob> &in_grad = outputs;

    Stream<gpu> *s = ctx.get_stream<gpu>();
    Init(in_data, s);

//    CHECK(ctx.is_train && !param_.use_global_stats)
//        << "use global statistics is not yet supported in CuDNNBatchNorm";

    DType *nhwc_X = GetNdPtr(in_data, dim_, s);
    DType *nhwc_dX = GetNdPtr(in_grad[nhwcbatchnorm::kData], dim_, s);
    DType *nhwc_Y = nullptr;
    DType *nhwc_dY = GetNdPtr(out_grad, dim_, s);
    nhwc_layer_->setInputOutputPointers(nhwc_X, nhwc_dX, nhwc_Y, nhwc_dY);

    int features = in_data.shape_[param_.axis];

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_gamma.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      Tensor<gpu, 1, DTypeParam> beta =
        in_beta.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      Tensor<gpu, 1, DTypeParam> dbeta =
        in_grad[nhwcbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> dgamma =
        in_grad[nhwcbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> save_mean =
        out_mean.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> save_inv_var =
        out_var.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      // beta.dptr_ only needed if fused relu.  OK to set always though.
      std::vector<void *> weights = {gamma.dptr_, beta.dptr_};
      std::vector<void *> dweights = {dgamma.dptr_, dbeta.dptr_};
      nhwc_layer_->setWeightPointers(weights, dweights);

      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);

      // MXNet already has allocated Tensors for the minibatch mean and variance.
      // These are the first two sizes returned by NhwcBatchNormSerialLayer::numWorkspaceBytes
      // The other sizes must be allocated explicitly as temp workspace tensors

      std::vector<void*> workspace;
      workspace.push_back(save_mean.dptr_);
      workspace.push_back(save_inv_var.dptr_);
      workspace.push_back(retired_ctas_hdl_.dptr);

      mshadow::Tensor<gpu, 1, ScaleType> temp_space =
          this->AllocateTempWorkspace(ctx, workspace_total_bytes_ + inuse_tempspace_bytes);

      for (auto offset : workspace_byte_offsets_) {
        void *dptr = reinterpret_cast<char *>(temp_space.dptr_) + inuse_tempspace_bytes + offset;
        workspace.push_back(dptr);
      }
      nhwc_layer_->setWorkspacePointers(workspace, workspace_bytes_);

      if (param_.fix_gamma) gamma = 1.f;
      bool fuse_relu = param_.act_type.has_value() &&
        (param_.act_type.value() == activation::kReLU);
      nhwc_layer_->setBNGroup(param_.bn_group, reinterpret_cast<void**>(param_.xbuf_ptr),
                              ctx.run_ctx.ctx.dev_id);
      nhwc_layer_->dgrad(Stream<gpu>::GetStream(s), fuse_relu, ctx.run_ctx.ctx.dev_id);
      if (param_.fix_gamma) dgamma = 0.f;
    })
  }

/*!
 * \brief Returns whether the nhwc kernel supports the batchnorm
 * operation described by `param`.
 */
  static bool Supports(const BatchNormParam &param, int dtype, const TShape& shape,
                       const Context& ctx) {
    // Axis parameters are in the range [0,shape.ndim()-1].
    using namespace mxnet::common::cuda;
    int dim = shape.ndim();
    bool retVal = !param.use_global_stats &&
           dim == 4 && dtype == mshadow::kFloat16 &&
           param.axis == 3 &&
           shape[3] % 4 == 0 &&
           SupportsCooperativeLaunch(ctx.dev_id);
    return retVal;
  }

  // Converts a TBlob to a dptr, checking for the expected dim and that it's contiguous.
  DType *GetNdPtr(const TBlob& tb, int dim, mshadow::Stream<gpu> *s) {
    DType *data_ptr = NULL;
    if (dim == 2) {
      mshadow::Tensor<gpu, 2, DType> data = tb.get<gpu, 2, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 3) {
      mshadow::Tensor<gpu, 3, DType> data = tb.get<gpu, 3, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 4) {
      mshadow::Tensor<gpu, 4, DType> data = tb.get<gpu, 4, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else if (dim == 5) {
      mshadow::Tensor<gpu, 5, DType> data = tb.get<gpu, 5, DType>(s);
      CHECK_EQ(data.CheckContiguous(), true);
      data_ptr = data.dptr_;
    } else {
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 2, 3, 4 or 5.";
    }
    return data_ptr;
  }

 private:
  void Init(const TBlob &in_data, mshadow::Stream<gpu> *s, const bool& is_train = true) {
      cudnnDataType_t     data_type     = CUDNN_DATA_HALF;
      cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NHWC;

      dim_ = in_data.ndim();
      CHECK_EQ(dim_, 4);
      CHECK_EQ(param_.axis, 3)
        << "axis must be 3, found: " << param_.axis;

      TShape shape_ = in_data.shape_;

      int n = shape_[0];
      int h = shape_[1];
      int w = shape_[2];
      int c = shape_[3];
      double exp_avg_factor = 1.0 - param_.momentum;
      double eps = param_.eps;
      int bn_group = is_train ? param_.bn_group : 1;

      nhwc_layer_->setInputDescriptor(tensor_format, data_type, n, c, h, w, bn_group);
      nhwc_layer_->setOutputDescriptor(tensor_format, data_type, n, c, h, w);

      // memory allocation - handled by MXNet

      // layer buffer setup.  Taken from nhwc_batchnorm_test.h
      nhwc_layer_->setConstants(exp_avg_factor, eps);

      // Check that first two returned values from numWorkspaceBytes are as expected
      workspace_bytes_ = nhwc_layer_->numWorkspaceBytes();
      auto expected_bytes = c * sizeof(ScaleType);
      CHECK_EQ(workspace_bytes_[0], expected_bytes) << "Unexpected save_mean workspace size.";
      CHECK_EQ(workspace_bytes_[1], expected_bytes) << "Unexpected save_variance workspace size.";

      // The next value from numWorkspaceBytes is the retired_ctas_ workspace.  This is not
      // a temp workspace either, because it needs a one-time initialization to 0.

      const int retired_cta_bytes = workspace_bytes_[2];
      if (!init_retired_ctas_) {
        retired_ctas_hdl_ = mxnet::Storage::Get()->Alloc(retired_cta_bytes, Context::GPU());
        // Zero the retired_ctas_ area once.  Kernels use this area to synchronize the kernel
        // blocks, which then leave the value as 0 for the next kernel.  This region can
        // be shared by all kernels launched by this thread, because they are launched into
        // the same stream and hence are serialized.
        CUDA_CALL(cudaMemsetAsync(retired_ctas_hdl_.dptr, 0,
          retired_cta_bytes, mshadow::Stream<gpu>::GetStream(s)));
        init_retired_ctas_ = true;
      }

      // Remainder of workspace_bytes are temp areas to be allocated.
      // This is done as one chunk, then carved up via separately offsetted pointers.
      int start_index = 3;
      workspace_byte_offsets_.clear();
      for (std::vector<size_t>::size_type i = start_index; i != workspace_bytes_.size(); ++i) {
         // First record the byte offset, which we desire to match default cudaMalloc alignment
         // of 512 bytes.
        workspace_total_bytes_ = round_up_to_multiple(workspace_total_bytes_, 512);
        workspace_byte_offsets_.push_back(workspace_total_bytes_);
        // Now increase the total bytes by this allocation
        auto alloc_bytes = workspace_bytes_[i];
        workspace_total_bytes_ += alloc_bytes;
      }
  }

  // Increase a value until it is a multiple of `multiple`.
  size_t round_up_to_multiple(size_t x, int multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
  }

  // Allocates a 1D Tensor of ScaleType words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  mshadow::Tensor<gpu, 1, ScaleType> AllocateTempWorkspace(const OpContext &ctx,
                                                           size_t size_bytes) {
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    size_t size_words = max(static_cast<size_t>(1),
                            (size_bytes + sizeof(ScaleType) - 1) / sizeof(ScaleType));
    auto retval = ctx.requested[nhwcbatchnorm::kTempSpace].get_space_typed<gpu, 1, ScaleType>(
        mshadow::Shape1(size_words), s);
    return retval;
  }

  std::shared_ptr<NhwcBatchNorm> nhwc_layer_;
  int dim_;
  cudnnDataType_t dtype_;
  int dtype_param_;
  BatchNormParam param_;
  std::vector<size_t> workspace_bytes_;
  size_t workspace_total_bytes_;
  std::vector<int> workspace_byte_offsets_;

  bool init_retired_ctas_ = false;
  mxnet::Storage::Handle retired_ctas_hdl_;
};
#endif  // defined(__CUDACC__)

#endif  // MXNET_USE_CUDNN == 1
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_NHWC_BATCH_NORM_INL_H_
