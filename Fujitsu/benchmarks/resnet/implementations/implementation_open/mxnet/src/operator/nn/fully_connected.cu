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
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
#if MXNET_USE_CUDA && CUDA_VERSION >= 8000
#include "./cublas_fully_connected-inl.h"
#endif

namespace mxnet {
namespace op {

#if MXNET_USE_CUDA && CUDA_VERSION >= 8000
template<typename DType>
static CuBLASFullyConnectedOp<DType> &GetCuBLASFullyConnectedOp(const FullyConnectedParam& param,
    const Context& ctx) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local CuBLASFullyConnectedOp<DType> op;
#else
  static MX_THREAD_LOCAL CuBLASFullyConnectedOp<DType> op;
#endif
  op.Init(param, ctx);
  return op;
}
#endif

template<>
void FullyConnectedCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), in_expected);
  CHECK_EQ(outputs.size(), 1U);
  int dtype = inputs[0].type_flag_;

#if MXNET_USE_CUDA && CUDA_VERSION >= 8000
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cublas_off || !CuBLASFullyConnectedOp<DType>::Supports(param, ctx.run_ctx.ctx)) {
      FCForward<gpu, DType>(ctx, param, inputs, req, outputs);
    } else {
      CuBLASFullyConnectedOp<DType> &op = GetCuBLASFullyConnectedOp<DType>(param, ctx.run_ctx.ctx);
      op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    FCForward<gpu, DType>(ctx, param, inputs, req, outputs);
  });
#endif
}

template<>
void FullyConnectedGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), out_expected);
  CHECK_EQ(req.size(), out_expected);

  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  int dtype = inputs[0].type_flag_;

#if MXNET_USE_CUDA && CUDA_VERSION >= 8000
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cublas_off || !CuBLASFullyConnectedOp<DType>::Supports(param, ctx.run_ctx.ctx)) {
      FCBackward<gpu, DType>(ctx, param, out_grad, in_data, req, outputs);
    } else {
      CuBLASFullyConnectedOp<DType> &op = GetCuBLASFullyConnectedOp<DType>(param, ctx.run_ctx.ctx);
      op.Backward(ctx, out_grad, in_data, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    FCBackward<gpu, DType>(ctx, param, out_grad, in_data, req, outputs);
  });
#endif
}

NNVM_REGISTER_OP(FullyConnected)
.set_attr<FCompute>("FCompute<gpu>", FullyConnectedCompute<gpu>);

NNVM_REGISTER_OP(_backward_FullyConnected)
.set_attr<FCompute>("FCompute<gpu>", FullyConnectedGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
