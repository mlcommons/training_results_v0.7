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
 * \file bn_stats_finalize-inl.h
 * \brief
 * \author Dick Carter
*/

#include "bn_stats_finalize-inl.h"
#include <nnvm/op_attr_types.h>
#include "../elemwise_op_common.h"
#include "../operator_common.h"

/*! \brief inverse standard deviation <-> variance */
#define VARIANCE_TO_INVSTD(__var$,    __eps$)   (1.0/std::sqrt((__var$) + DType(__eps$)))
#define INVSTD_TO_VARIANCE(__invstd$, __eps$)   ((1.0 / ((__invstd$) * (__invstd$))) - (__eps$))

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(BNStatsFinalizeParam);

static bool BNStatsFinalizeShape(const nnvm::NodeAttrs& attrs,
                           std::vector<mxnet::TShape> *in_shape,
                           std::vector<mxnet::TShape> *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), static_cast<size_t>(bn_stats_finalize::kNumInputs))
    << "Input:[sum, sum_squares, gamma, beta, MovingMean, MovingVar]";
  CHECK_EQ(out_shape->size(), static_cast<size_t>(bn_stats_finalize::kNumOutputs));
  const mxnet::TShape &sum_shape = in_shape->at(bn_stats_finalize::kSum);

  if (sum_shape.ndim() == 0) {
    return false;
  }

  SHAPE_ASSIGN_CHECK(*in_shape, bn_stats_finalize::kSumOfSquares, sum_shape);
  SHAPE_ASSIGN_CHECK(*in_shape, bn_stats_finalize::kGamma, sum_shape);
  SHAPE_ASSIGN_CHECK(*in_shape, bn_stats_finalize::kBeta, sum_shape);
  SHAPE_ASSIGN_CHECK(*in_shape, bn_stats_finalize::kInMovingMean, sum_shape);  // kMovingMean
  SHAPE_ASSIGN_CHECK(*in_shape, bn_stats_finalize::kInMovingVar, sum_shape);  // kMovingVar

  out_shape->clear();
  // All outputs have the same shape
  for (int i = 0; i != bn_stats_finalize::kNumOutputs; ++i)
    out_shape->push_back(sum_shape);

  return true;
}

static bool BNStatsFinalizeType(const nnvm::NodeAttrs& attrs,
                          std::vector<int> *in_type, std::vector<int> *out_type) {
  using namespace mshadow;
  int dtype = out_type->size() == 0 ? mshadow::kFloat16 : (*out_type)[0];
  // It may be hard to discern diffent flavors of BNStatsFinalize that differ only in their output
  // type (e.g. float16 vs. float32).  For now assume output dtype float16 for all input dtypes.
  if (dtype == -1)
    dtype = mshadow::kFloat16;
  // For float16 equiv_scale and equiv_bias output types,
  // beta, gamma, mean, and average are stored in float32.
  // For other output types, these parameters have the same type as input
  int dtype_param;
  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_param = mshadow::DataType<AccRealX>::kFlag; });
  std::vector<std::string> args{"sum", "sum_squares", "gamma", "beta", "mean", "var"};
  CHECK_LE(in_type->size(), args.size());
  for (size_t i = 0; i < in_type->size(); ++i) {
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = dtype_param;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], dtype_param, args[i]);
    }
  }
  out_type->clear();
  // First two outputs (equiv_scale and equiv_bias) are dtype, rest are of type dtype_param
  out_type->push_back(dtype);  // equiv_scale
  out_type->push_back(dtype);  // equiv_bias
  while (out_type->size() < static_cast<size_t>(bn_stats_finalize::kNumOutputs))
    out_type->push_back(dtype_param);
  return true;
}


static inline bool BNStatsFinalizeStorageType(const nnvm::NodeAttrs &attrs,
                                        const int dev_mask,
                                        DispatchMode *dispatch_mode,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  const BNStatsFinalizeParam &param = nnvm::get<BNStatsFinalizeParam>(attrs.parsed);

  bool dispatched = false;
  for (int& v : *in_attrs)
    if (v == - 1) v = kDefaultStorage;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  if (!common::ContainsOnlyStorage(*in_attrs, kDefaultStorage) && param.fix_gamma) {
    LOG(FATAL) << "fix_gamma=True is not supported for sparse ndarrays. Tracked at #11647";
  }
  return dispatched;
}

std::vector<nnvm::NodeEntry> BNStatsFinalizeGrad(const nnvm::NodePtr& n,
                                           const std::vector<nnvm::NodeEntry>& ograds) {
  const BNStatsFinalizeParam &param = nnvm::get<BNStatsFinalizeParam>(n->attrs.parsed);
  CHECK(param.output_mean_var) << "Expecting output_mean_var true on training graph";
  std::vector<nnvm::NodeEntry> out_data(n->num_outputs());
  for (uint32_t i = 0; i < out_data.size(); ++i) {
    out_data[i] = nnvm::NodeEntry{n, i, 0};
  }
  std::vector<nnvm::NodeEntry> heads;
  // Only incoming gradients are inputs of the backward node.
  // Technically, d_equiv_scale and d_equiv_bias are not needed, but included for simplicity
  // of index calculation.
  CHECK_EQ(ograds.size(), static_cast<size_t>(bn_stats_finalize::kNumOutputs))
    << "Incorrect number of incoming gradients of BNStatsFinalizeParam node";
  heads.reserve(bn_stats_finalize::kNumOutputs);
  for (int i = 0; i != bn_stats_finalize::kNumOutputs; ++i)
    heads.push_back(ograds[i]);

  nnvm::NodePtr gnode = nnvm::Node::Create();
  gnode->inputs = std::move(heads);
  gnode->control_deps.emplace_back(n);
  gnode->attrs = n->attrs;
  gnode->attrs.op = nnvm::Op::Get("_backward_BNStatsFinalize");
  gnode->attrs.name = n->attrs.name + "_backward";
  // Prepare a no-gradient node to forbid gradient on aux_states
  nnvm::NodePtr ng = nnvm::Node::Create();
  ng->attrs.op = Op::Get("_NoGradient");
  ng->attrs.name = "NoGradient";
  // The set the vector of input gradients
  std::vector<nnvm::NodeEntry> in_grad(bn_stats_finalize::kNumInputs);
  for (uint32_t i = 0; i < bn_stats_finalize::kNumInputs; ++i) {
    in_grad[i] = (i < bn_stats_finalize::kNumNonAuxInputs) ? nnvm::NodeEntry{gnode, i, 0}
                                                           : nnvm::NodeEntry{ng, 0, 0};
  }
  return in_grad;
}

NNVM_REGISTER_OP(BNStatsFinalize)
.describe(R"code(Batch normalization stats finalize.

This is an experimental operator designed to work in concert with the NormalizedConvolution op.
Think of batchnorm as split into:
  1) input data statistics generation (but now sum and sum_squares, not mean and variance)
  2) statistics finalize (maps sum, sum_squares, beta and gamma to an equiv_scale and equiv_bias)
  3) apply equiv_scale and equiv_bias to data

With this picture, the NormalizedConvolution includes parts 1) and 3) from above:

NormalizedConvolution == StatsApply -> Relu -> Convolution -> StatsGen

What's left over from this NormalizedConvolution is BNStatsFinalize, which performs the mapping
of part 2) above, plus the running mean, running variance state machine update of Batchnorm.

Legacy description of Batchnorm:

Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
well as offset ``beta``.

Assume the input has more than one dimension and we normalize along axis 1.
We first compute the mean and variance along this axis:

.. math::

  data\_mean[i] = mean(data[:,i,:,...]) \\
  data\_var[i] = var(data[:,i,:,...])

Then compute the normalized output, which has the same shape as input, as following:

.. math::

  out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

Both *mean* and *var* returns a scalar by treating the input as a vector.

Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these
two outputs are blocked.

Besides the inputs and the outputs, this operator accepts two auxiliary
states, ``moving_mean`` and ``moving_var``, which are *k*-length
vectors. They are global statistics for the whole dataset, which are updated
by::

  moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  moving_var = moving_var * momentum + data_var * (1 - momentum)

If ``use_global_stats`` is set to be true, then ``moving_mean`` and
``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
the output. It is often used during inference.

The parameter ``axis`` specifies which axis of the input shape denotes
the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
axis to be the last item in the input shape.

Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
then set ``gamma`` to 1 and its gradient to 0.

.. Note::
  When ``fix_gamma`` is set to True, no sparse support is provided. If ``fix_gamma is`` set to False,
  the sparse tensors will fallback.

)code" ADD_FILELINE)
.set_num_inputs(bn_stats_finalize::kNumInputs)
.set_num_outputs(bn_stats_finalize::kNumOutputs)
.set_attr_parser(ParamParser<BNStatsFinalizeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"sum", "sum_squares", "gamma", "beta",
                                  "moving_mean", "moving_var"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"equiv_scale", "equiv_bias", "mean", "var",
                                  "gamma_out", "beta_out"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const BNStatsFinalizeParam& param = nnvm::get<BNStatsFinalizeParam>(attrs.parsed);
  return param.output_mean_var ? bn_stats_finalize::kNumOutputs : 2;
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  return std::vector<uint32_t>{bn_stats_finalize::kInMovingMean, bn_stats_finalize::kInMovingVar};
})
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  // gamma and beta can share storage with gamma_out and beta_out respectively.
  return std::vector<std::pair<int, int>>{{bn_stats_finalize::kGamma, bn_stats_finalize::kGammaOut},
                                          {bn_stats_finalize::kBeta, bn_stats_finalize::kBetaOut}};
})
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity", [](const NodeAttrs& attrs){
  // Pass gamma and beta forward unaltered.
  return std::vector<bool>{true, true};
})
.set_attr<mxnet::FInferShape>("FInferShape", BNStatsFinalizeShape)
.set_attr<nnvm::FInferType>("FInferType", BNStatsFinalizeType)
.set_attr<FInferStorageType>("FInferStorageType", BNStatsFinalizeStorageType)
.set_attr<FCompute>("FCompute<cpu>", BNStatsFinalizeCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", BNStatsFinalizeGrad)
.add_argument("sum", "NDArray-or-Symbol", "sum of input data to be normalized")
.add_argument("sum_squares", "NDArray-or-Symbol", "sum of squares of input data to be normalized")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(BNStatsFinalizeParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == bn_stats_finalize::kInMovingMean) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == bn_stats_finalize::kInMovingVar) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

NNVM_REGISTER_OP(_backward_BNStatsFinalize)
.set_num_inputs(bn_stats_finalize::kNumOutputs)
.set_num_outputs(bn_stats_finalize::kNumNonAuxInputs)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption", [](const NodeAttrs& attrs){
  // Pass output gradients coming in for 'gamma' and 'beta' untouched.
  return std::vector<std::pair<int, int>>{{bn_stats_finalize::kGammaOut, bn_stats_finalize::kGamma},
                                          {bn_stats_finalize::kBetaOut, bn_stats_finalize::kBeta}};
})
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity", [](const NodeAttrs& attrs){
  // Passed back gradients are unaltered.
  return std::vector<bool>{true, true};
})
.set_attr<FInferStorageType>("FInferStorageType", BNStatsFinalizeStorageType)
.set_attr_parser(ParamParser<BNStatsFinalizeParam>)
.set_attr<FCompute>("FCompute<cpu>", BNStatsFinalizeGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
