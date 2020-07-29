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
 * Copyright (c) 2017 by Contributors
 * \file normalized_convolution.cu
 * \brief
 * \author Dick Carter
*/

#include "./normalized_convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_normalized_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNNormalizedConvolutionOp<DType>& GetCuDNNNormalizedConvOp(
                                                 const NormalizedConvolutionParam& param,
                                                 bool output_stats,
                                                 const std::vector<mxnet::TShape>& in_shape,
                                                 const std::vector<mxnet::TShape>& out_shape,
                                                 const OpContext& ctx) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<NormalizedConvSignature,
                                         std::shared_ptr<CuDNNNormalizedConvolutionOp<DType> >,
                                         OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<NormalizedConvSignature,
                                            std::shared_ptr<CuDNNNormalizedConvolutionOp<DType> >,
                                            OpHash> ops;
#endif
  NormalizedConvSignature key(param);
  size_t ndim = 0;
  for (auto &s : in_shape)
    ndim += s.ndim();
  for (auto &s : out_shape)
    ndim += s.ndim();
  key.Reserve(1 /* for output_stats */ +
              ndim /* for in and out shapes */ +
              1 /* for dev_id */);

  key.AddSign(output_stats);
  key.AddSign(in_shape);
  key.AddSign(out_shape);
  key.AddSign(ctx.run_ctx.ctx.dev_id);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuDNNNormalizedConvolutionOp<DType>> op(
        new CuDNNNormalizedConvolutionOp<DType>());
    auto ins_ret = ops.insert(std::pair<NormalizedConvSignature,
        std::shared_ptr<CuDNNNormalizedConvolutionOp<DType>>>(key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(param, output_stats, in_shape, out_shape, ctx);
  }
  return *it->second;
}
#endif

template<>
void NormalizedConvolutionCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const NormalizedConvolutionParam& param = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  int dtype = inputs[normalized_conv::kData].type_flag_;
  mxnet::TShape in_data_shape = inputs[normalized_conv::kData].shape_;
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (!CuDNNNormalizedConvolutionOp<DType>::Supports(param, in_data_shape,
                                                       ctx.run_ctx.ctx.dev_id)) {
      LOG(WARNING) << "This NormalizedConvolution is not supported by cudnn"
                   << ", MXNET NormalizedConvolution is applied.";
      NormalizedConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs, req, outputs);
    } else {
      std::vector<mxnet::TShape> in_shape(inputs.size());
      std::vector<mxnet::TShape> out_shape(outputs.size());
      bool output_stats = CuDNNNormalizedConvolutionOp<DType>::OutputStats(ctx, req);
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = inputs[i].shape_;
      for (size_t i = 0; i < out_shape.size(); i++)
        out_shape[i] = outputs[i].shape_;
      CuDNNNormalizedConvolutionOp<DType> &op = GetCuDNNNormalizedConvOp<DType>(param,
        output_stats, in_shape, out_shape, ctx);
      op.Forward(ctx, inputs, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    NormalizedConvolutionOp<gpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

template<>
void NormalizedConvolutionGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const NormalizedConvolutionParam& param = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  size_t num_fwd_inputs = normalized_conv::NumInputs(param.no_equiv_scale_bias);
  size_t num_fwd_ios = inputs.size();
  size_t num_fwd_outputs = num_fwd_ios - num_fwd_inputs;
  std::vector<TBlob> fwd_out_data(inputs.begin(), inputs.begin() + num_fwd_outputs);
  std::vector<TBlob> fwd_in_data(inputs.begin() + num_fwd_outputs, inputs.end());
  // Remember, for fwd_out_data[kOut], we've swapped in the gradient for the output itself.
  const TBlob &out_grad = fwd_out_data[normalized_conv::kOut];
  // Gradient types will be the same as the corresponding output
  int dtype = out_grad.type_flag_;
  mxnet::TShape in_data_shape = fwd_in_data[normalized_conv::kData].shape_;
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (!CuDNNNormalizedConvolutionOp<DType>::Supports(param, in_data_shape,
                                                       ctx.run_ctx.ctx.dev_id)) {
      LOG(WARNING) << "This NormalizedConvolution is not supported by cudnn"
                   << ", MXNET NormalizedConvolution is applied.";
      NormalizedConvolutionOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, fwd_in_data, req, outputs);
    } else {
      std::vector<mxnet::TShape> in_shape(fwd_in_data.size());
      std::vector<mxnet::TShape> out_shape(fwd_out_data.size());
      for (size_t i = 0; i < in_shape.size(); i++)
        in_shape[i] = fwd_in_data[i].shape_;
      for (size_t i = 0; i < out_shape.size(); i++)
        out_shape[i] = fwd_out_data[i].shape_;
      // The Backward() call performs the same function, regardless of 'output_stats', so in that
      // sense, the setting is arbitrary.  However, all configurations of NormalizedConvolution
      // will have a 'false' version after they have gone through the validation step.  Since
      // we can't call the OutputStats(ctx, req) routine since 'req' refers to the forward outputs,
      // we just assume 'false' for simplicity:
      bool output_stats = false;
      CuDNNNormalizedConvolutionOp<DType> &op = GetCuDNNNormalizedConvOp<DType>(param,
        output_stats, in_shape, out_shape, ctx);
      op.Backward(ctx, std::vector<TBlob>{out_grad}, fwd_in_data, req, outputs);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    NormalizedConvolutionOp<gpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, fwd_in_data, req, outputs);
  })
#endif  // MXNET_USE_CUDNN
}

NNVM_REGISTER_OP(NormalizedConvolution)
.set_attr<FCompute>("FCompute<gpu>", NormalizedConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_NormalizedConvolution)
.set_attr<FCompute>("FCompute<gpu>", NormalizedConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

