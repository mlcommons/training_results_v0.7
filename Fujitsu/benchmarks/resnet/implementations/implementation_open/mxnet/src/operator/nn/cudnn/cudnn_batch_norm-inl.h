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
 * \file cudnn_batch_norm-inl.h
 * \brief
 * \author Junyuan Xie
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_INL_H_
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "../batch_norm-inl.h"

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1
namespace cudnnbatchnorm {
enum CuDNNBatchNormOpInputs {kData, kGamma, kBeta};
enum CuDNNBatchNormOpOutputs {kOut, kMean, kInvVar};
enum CuDNNBatchNormOpAuxiliary {kMovingMean, kMovingInvVar};
}  // namespace cudnnbatchnorm

#if defined(__CUDACC__)
template<typename DType>
class CuDNNBatchNormOp {
  STATIC_ASSERT_CUDNN_VERSION_GE(5000);

 public:
  CuDNNBatchNormOp() {
    using namespace mshadow;
    dtype_ = DataType<DType>::kCudnnFlag;
    // For float16 input type beta, gamma, mean, and average are stored in float32.
    // For other input types, these parameters have the same type as input
    dtype_param_ = (dtype_ == CUDNN_DATA_HALF) ? kFloat32 : DataType<DType>::kFlag;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&io_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&mean_desc_));
  }

  void Init(const BatchNormParam &param) {
    CHECK_GE(param.eps, CUDNN_BN_MIN_EPSILON)
     << "CuDNN requires eps to be no less than " << CUDNN_BN_MIN_EPSILON;
    this->param_ = param;
  }

  ~CuDNNBatchNormOp() {
    CUDNN_CALL(cudnnDestroyTensorDescriptor(io_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(mean_desc_));
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
    CHECK_EQ(req[cudnnbatchnorm::kOut], kWriteTo);
    CHECK_GE(in_data[cudnnbatchnorm::kData].ndim(), 2);
    CHECK_LE(in_data[cudnnbatchnorm::kData].ndim(), 5);

    Init(in_data[cudnnbatchnorm::kData]);
    int features = in_data[cudnnbatchnorm::kData].shape_[param_.axis];

    Stream<gpu> *s = ctx.get_stream<gpu>();
    DType *x_dptr = GetNdPtr(in_data[cudnnbatchnorm::kData], dim_, s);
    DType *y_dptr = GetNdPtr(out_data[cudnnbatchnorm::kOut], dim_, s);
#if CUDNN_VERSION >= 7002
    auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    auto mode = CUDNN_BATCHNORM_SPATIAL;
#endif

    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_data[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> beta =
        in_data[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> moving_mean =
        aux_states[cudnnbatchnorm::kMovingMean]
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> moving_inv_var =
        aux_states[cudnnbatchnorm::kMovingInvVar]
        .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      typename DataType<DType>::ScaleType a = 1.0f;
      typename DataType<DType>::ScaleType b = 0.0f;

      if (param_.fix_gamma) gamma = 1.f;

      if (ctx.is_train) {
        Tensor<gpu, 1, DTypeParam> save_mean =
          out_data[cudnnbatchnorm::kMean].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        Tensor<gpu, 1, DTypeParam> save_inv_var =
          out_data[cudnnbatchnorm::kInvVar]
          .get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
        CUDNN_CALL(cudnnBatchNormalizationForwardTraining(s->dnn_handle_,
                                                          mode,
                                                          &a,
                                                          &b,
                                                          io_desc_,
                                                          x_dptr,
                                                          io_desc_,
                                                          y_dptr,
                                                          mean_desc_,
                                                          gamma.dptr_,
                                                          beta.dptr_,
                                                          1 - param_.momentum,
                                                          moving_mean.dptr_,
                                                          moving_inv_var.dptr_,
                                                          param_.eps,
                                                          save_mean.dptr_,
                                                          save_inv_var.dptr_));
      } else {
        CUDNN_CALL(cudnnBatchNormalizationForwardInference(s->dnn_handle_,
                                                           CUDNN_BATCHNORM_SPATIAL,
                                                           &a,
                                                           &b,
                                                           io_desc_,
                                                           x_dptr,
                                                           io_desc_,
                                                           y_dptr,
                                                           mean_desc_,
                                                           gamma.dptr_,
                                                           beta.dptr_,
                                                           moving_mean.dptr_,
                                                           moving_inv_var.dptr_,
                                                           param_.eps));
      }
    })
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &outputs) {
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
    const std::vector<TBlob> &in_grad = outputs;

    Init(in_data);
    Stream<gpu> *s = ctx.get_stream<gpu>();

    DType *x_dptr = GetNdPtr(in_data, dim_, s);
    DType *dx_dptr = GetNdPtr(in_grad[cudnnbatchnorm::kData], dim_, s);
    DType *dy_dptr = GetNdPtr(out_grad, dim_, s);

    const bool global_stats = !ctx.is_train || param_.use_global_stats;

    int features = in_data.shape_[param_.axis];

#if CUDNN_VERSION >= 7002
    auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    auto mode = CUDNN_BATCHNORM_SPATIAL;
#endif
    MSHADOW_REAL_TYPE_SWITCH(dtype_param_, DTypeParam, {
      Tensor<gpu, 1, DTypeParam> gamma =
        in_gamma.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> dbeta =
        in_grad[cudnnbatchnorm::kBeta].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> dgamma =
        in_grad[cudnnbatchnorm::kGamma].get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> save_mean =
        out_mean.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);
      Tensor<gpu, 1, DTypeParam> save_inv_var =
        out_var.get_with_shape<gpu, 1, DTypeParam>(Shape1(features), s);

      typename DataType<DType>::ScaleType a = 1.0f;
      typename DataType<DType>::ScaleType b = 0.0f;
      typename DataType<DType>::ScaleType b_add = 1.0f;
      CHECK_EQ(s->dnn_handle_ownership_, mshadow::Stream<gpu>::OwnHandle);

      if (param_.fix_gamma) gamma = 1.f;

      CUDNN_CALL(cudnnBatchNormalizationBackward(
        s->dnn_handle_,
        mode,
        &a,
        &b,
        &a,
        req[cudnnbatchnorm::kGamma] == kWriteTo ? &b: &b_add,
        io_desc_,
        x_dptr,
        io_desc_,
        dy_dptr,
        io_desc_,
        dx_dptr,
        mean_desc_,
        gamma.dptr_,
        dgamma.dptr_,
        dbeta.dptr_,
        param_.eps,
        global_stats ? nullptr : save_mean.dptr_,
        global_stats ? nullptr : save_inv_var.dptr_));
      if (param_.fix_gamma) dgamma = 0.f;
    })
  }

/*!
 * \brief Returns whether the cuDNN library version supports the batchnorm
 * operation described by `param`.
 */
  static bool Supports(const BatchNormParam &param, int dtype, const TShape& shape) {
    // Axis parameters are in the range [0,shape.ndim()-1].
    int dim = shape.ndim();
    return !param.use_global_stats &&
           dim >= 2 && dim <= 5 &&
           (param.axis == 1 || param.axis == dim-1);
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
      LOG(FATAL) << "Unexpected Tensor size " << dim << ", supporting only 3, 4 or 5.";
    }
    return data_ptr;
  }

 private:
  void Init(const TBlob &in_data) {
    // From merge
    dim_ = in_data.ndim();
    TShape in_shape = in_data.shape_;
    CHECK(param_.axis == 1 || param_.axis == dim_-1)
      << "axis must be 1 or " << (dim_-1) << ", found: " << param_.axis;
    bool transposed_layout = (dim_ >= 3 && param_.axis == dim_-1);
    auto format = transposed_layout ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    int n = in_shape[0];
    int c = transposed_layout ? in_shape[dim_ - 1] : in_shape[1];
    int w_index = transposed_layout ? dim_ - 2 : dim_ - 1;
    int d = dim_ >= 5 ? in_shape[w_index - 2] : 1;
    int h = dim_ >= 4 ? in_shape[w_index - 1] : 1;
    int w = dim_ >= 3 ? in_shape[w_index - 0] : 1;
     if (dim_ < 5) {
      CUDNN_CALL(cudnnSetTensor4dDescriptor(io_desc_,
                                            format,
                                            dtype_,
                                            n, c, h, w));
    } else {
      std::vector<int> ishape = {n, c, d, h, w};
      std::vector<int> istride = {c * d * h * w,
                                      d * h * w,
                                          h * w,
                                              w,
                                              1};
      if (transposed_layout)
        istride = {c * d * h * w,
                               1,
                       c * h * w,
                           c * w,
                               c};
      CUDNN_CALL(cudnnSetTensorNdDescriptor(io_desc_,
                                            dtype_,
                                            dim_,
                                            &ishape[0],
                                            &istride[0]));
    }
    CUDNN_CALL(cudnnDeriveBNTensorDescriptor(mean_desc_,
                                             io_desc_,
                                             CUDNN_BATCHNORM_SPATIAL));
  }

  int dim_;
  cudnnDataType_t dtype_;
  int dtype_param_;
  cudnnTensorDescriptor_t io_desc_, mean_desc_;
  BatchNormParam param_;
};
#endif  // defined(__CUDACC__)

#endif  // MXNET_USE_CUDNN == 1
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_INL_H_
