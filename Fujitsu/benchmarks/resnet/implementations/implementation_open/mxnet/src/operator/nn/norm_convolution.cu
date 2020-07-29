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
 * \file norm_convolution.cu
 * \brief
 * \author Dick Carter
*/

#include "./norm_convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_norm_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNNormConvolutionOp<DType>& GetCuDNNNormConvOp(
                                                 const NormConvolutionParam& param,
                                                 bool output_stats,
                                                 const std::vector<mxnet::TShape> &in_shapes,
                                                 const mxnet::TShape &out_shape,
                                                 const OpContext& ctx) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<NormConvSignature,
                                         std::shared_ptr<CuDNNNormConvolutionOp<DType> >,
                                         OpHash> ops;
#else
  static MX_THREAD_LOCAL std::unordered_map<NormConvSignature,
                                            std::shared_ptr<CuDNNNormConvolutionOp<DType> >,
                                            OpHash> ops;
#endif
  NormConvSignature key(param);
  size_t input_ndim = 0;
  for (auto &s : in_shapes)
    input_ndim += s.ndim();
  key.Reserve(1 /* for output_stats */ +
              input_ndim /* for primary input data shape */ +
              out_shape.ndim() /* for primary output data shape */ +
              1 /* for dev_id */);

  key.AddSign(output_stats);
  key.AddSign(in_shapes);
  key.AddSign(out_shape);
  key.AddSign(ctx.run_ctx.ctx.dev_id);

  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuDNNNormConvolutionOp<DType>> op(
        new CuDNNNormConvolutionOp<DType>());
    auto ins_ret = ops.insert(std::pair<NormConvSignature,
        std::shared_ptr<CuDNNNormConvolutionOp<DType>>>(key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
    it->second->Init(param, output_stats, in_shapes, out_shape, ctx);
  }
  return *it->second;
}
#endif

template<>
void NormConvolutionCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const NormConvolutionParam& param = nnvm::get<NormConvolutionParam>(attrs.parsed);
  int dtype = inputs[norm_conv::kData].type_flag_;
  mxnet::TShape in_data_shape = inputs[norm_conv::kData].shape_;
  mxnet::TShape out_data_shape = outputs[norm_conv::kOut].shape_;
  CHECK_EQ(inputs.size(), static_cast<size_t>(norm_conv::NumInputs(param.no_norm)));
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (CuDNNNormConvolutionOp<DType>::Supports(param, in_data_shape,
                                                       ctx.run_ctx.ctx.dev_id)) {
      std::vector<mxnet::TShape> in_shapes(inputs.size());
      for (size_t i = 0; i < in_shapes.size(); i++)
        in_shapes[i] = inputs[i].shape_;
      bool output_stats = CuDNNNormConvolutionOp<DType>::OutputStats(ctx, req);
      CuDNNNormConvolutionOp<DType> &op = GetCuDNNNormConvOp<DType>(param,
        output_stats, in_shapes, out_data_shape, ctx);
      op.Forward(ctx, inputs, req, outputs);
    } else {
      LOG(FATAL) << "No fallback impl for unsupported NormConvolution configuration.";
    }
  })
#else
  LOG(FATAL) << "Only cudnn-based NormConvolution supported.";
#endif  // MXNET_USE_CUDNN
}

template<>
void NormConvolutionGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  const NormConvolutionParam& param = nnvm::get<NormConvolutionParam>(attrs.parsed);
  size_t num_fwd_inputs = norm_conv::NumInputs(param.no_norm);
  size_t num_output_gradients = 1;  // only dOut gradient needed
  size_t num_fwd_ios = inputs.size();
  size_t num_fwd_outputs = num_fwd_ios - num_fwd_inputs - num_output_gradients;
  std::vector<TBlob> out_grads(inputs.begin(), inputs.begin() + num_output_gradients);
  std::vector<TBlob> fwd_out_data(inputs.begin() + num_output_gradients,
                                  inputs.begin() + num_output_gradients + num_fwd_outputs);
  std::vector<TBlob> fwd_in_data(inputs.begin() + num_output_gradients + num_fwd_outputs,
                                 inputs.end());
  // Remember, for fwd_out_data[kOut], we've swapped in the gradient for the output itself.
  const TBlob &out_grad = out_grads[norm_conv::kOut];
  // Gradient types will be the same as the corresponding output
  int dtype = out_grad.type_flag_;
  mxnet::TShape in_data_shape = fwd_in_data[norm_conv::kData].shape_;
  mxnet::TShape out_data_shape = out_grads[norm_conv::kOut].shape_;
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (CuDNNNormConvolutionOp<DType>::Supports(param, in_data_shape,
                                                       ctx.run_ctx.ctx.dev_id)) {
      // The Backward() call performs the same function, regardless of 'output_stats', so in that
      // sense, the setting is arbitrary.  However, all configurations of NormConvolution
      // will have a 'false' version after they have gone through the validation step.  Since
      // we can't call the OutputStats(ctx, req) routine since 'req' refers to the forward outputs,
      // we just assume 'false' for simplicity:
      bool output_stats = false;
      std::vector<mxnet::TShape> in_shapes(fwd_in_data.size());
      for (size_t i = 0; i < fwd_in_data.size(); i++)
        in_shapes[i] = fwd_in_data[i].shape_;
      CuDNNNormConvolutionOp<DType> &op = GetCuDNNNormConvOp<DType>(param,
        output_stats, in_shapes, out_data_shape, ctx);
      op.Backward(ctx, out_grads, fwd_in_data, fwd_out_data, req, outputs);
    } else {
      LOG(FATAL) << "No fallback impl for unsupported NormConvolution configuration.";
    }
  })
#else
  LOG(FATAL) << "Only cudnn-based NormConvolution supported.";
#endif  // MXNET_USE_CUDNN
}

namespace norm_conv {

// Can the function of this convolution op node be handled by norm convolution?
bool IsCompatibleConvolution(const nnvm::NodePtr& node, const int& dtype,
                             const TShape& shape, const Context& ctx) {
  auto param = nnvm::get<ConvolutionParam>(node->attrs.parsed);
  bool is_compatible = false;
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    // TODO(cfujitsang): add condition if using very specific Convolution parameters
    //                   example: cudnn_algo_fwd
    is_compatible = param.no_bias &&
                    CuDNNNormConvolutionOp<DType>::Supports(param, shape, ctx.dev_id);
  });
#endif  // MXNET_USE_CUDNN
  return is_compatible;
}

}  // namespace norm_conv

NNVM_REGISTER_OP(NormConvolution)
.set_attr<FCompute>("FCompute<gpu>", NormConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_NormConvolution)
.set_attr<FCompute>("FCompute<gpu>", NormConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

