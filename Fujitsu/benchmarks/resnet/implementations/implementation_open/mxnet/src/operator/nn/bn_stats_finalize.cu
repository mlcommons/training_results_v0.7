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
 * Copyright (c) 2019 by Contributors
 * \file bn_stats_finalize.cu
 * \brief Batch Normalization Stats Finalize code
 * \author Dick Carter
*/
#include <cuda_runtime_api.h>
#include <algorithm>
#include "batch_norm-inl.h"

#include "bn_stats_finalize-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_bn_stats_finalize-inl.h"
#endif

#include "../../common/cuda_utils.h"
#include "../../../include/mxnet/tensor_blob.h"

using namespace mxnet;

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNBNStatsFinalizeOp<DType>& GetCuDNNBNStatsFinalizeOp(
    const BNStatsFinalizeParam& param,
    const TShape& shape,
    const OpContext& ctx) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<BNStatsFinalizeSignature,
  std::shared_ptr<CuDNNBNStatsFinalizeOp<DType> >,
      OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<BNStatsFinalizeSignature,
                                        std::shared_ptr<CuDNNBNStatsFinalizeOp<DType> >,
                                        OpHash> ops;
#endif
  BNStatsFinalizeSignature key(param);
  key.Reserve(shape.ndim());
  key.AddSign(shape);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuDNNBNStatsFinalizeOp<DType>> op(
        new CuDNNBNStatsFinalizeOp<DType>());
    auto ins_ret = ops.insert(std::pair<BNStatsFinalizeSignature,
        std::shared_ptr<CuDNNBNStatsFinalizeOp<DType>>>(key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(param, shape, ctx);
  }
  return *it->second;
}
#endif

template<>
void BNStatsFinalizeCompute<gpu>(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), static_cast<size_t>(bn_stats_finalize::kNumInputs));
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + bn_stats_finalize::kNumNonAuxInputs);
  std::vector<TBlob> aux_states(inputs.begin() + bn_stats_finalize::kNumNonAuxInputs, inputs.end());
  // Note that the equiv_scale and equiv_bias outputs are float16, with the other i/o's float32
  int dtype = outputs[0].type_flag_;
  TShape shape = outputs[0].shape_;

#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  BNStatsFinalizeParam param = nnvm::get<BNStatsFinalizeParam>(attrs.parsed);
  // We enable the discrete NHWC cuda kernels by the same 'cudnn_off' flag.
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
    if (CuDNNBNStatsFinalizeOp<DType>::Supports(param, dtype, shape))
      GetCuDNNBNStatsFinalizeOp<DType>(param, shape, ctx).Forward(ctx, in_data, req, outputs,
                                                                  aux_states);
    else
      LOG(FATAL) << "No fallback impl for unsupported BNStatsFinalize configuration.";
  });
#else
  LOG(FATAL) << "Only cudnn-based BNStatsFinalize supported.";
#endif
}

template<>
void BNStatsFinalizeGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                               const OpContext& ctx, const std::vector<TBlob>& inputs,
                               const std::vector<OpReqType>& req,
                               const std::vector<TBlob>& outputs) {
  // Only incoming gradients (== number of fwd outputs) are inputs of the backward node.
  CHECK_EQ(inputs.size(), static_cast<size_t>(bn_stats_finalize::kNumOutputs));
  // Learn of the dtype of the backward node from an otherwise-ignored d_equiv_scale
  int dtype = inputs[0].type_flag_;
  TShape shape = inputs[0].shape_;

#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  BNStatsFinalizeParam param = nnvm::get<BNStatsFinalizeParam>(attrs.parsed);
  // We enable the discrete NHWC cuda kernels by the same 'cudnn_off' flag.
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
    if (CuDNNBNStatsFinalizeOp<DType>::Supports(param, dtype, shape))
      GetCuDNNBNStatsFinalizeOp<DType>(param, shape, ctx).Backward(ctx, inputs, req, outputs);
    else
      LOG(FATAL) << "No fallback impl for unsupported BNStatsFinalize configuration.";
  });
#else
  LOG(FATAL) << "Only cudnn-based BNStatsFinalize supported.";
#endif
}

NNVM_REGISTER_OP(BNStatsFinalize)
.set_attr<FCompute>("FCompute<gpu>", BNStatsFinalizeCompute<gpu>);

NNVM_REGISTER_OP(_backward_BNStatsFinalize)
.set_attr<FCompute>("FCompute<gpu>", BNStatsFinalizeGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
