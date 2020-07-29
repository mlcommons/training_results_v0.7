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
 * \file norm_convolution.cc
 * \brief
 * \author Dick Carter
*/

#include "./norm_convolution-inl.h"
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#if MXNET_USE_NNPACK == 1
#include "../nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(NormConvolutionParam);

static inline index_t AddPad(index_t dsize, index_t pad) {
  return dsize + 2 * pad;
}

static inline std::vector<std::string> ListArguments(const NormConvolutionParam& param_) {
  if (!param_.no_norm) {
    return {"data", "in_sum", "in_sum_squares", "gamma", "beta",
            "moving_mean", "moving_var", "weight"};
  } else {
    return {"data", "weight"};
  }
}

static bool NormConvolutionShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  using namespace mshadow;
  const NormConvolutionParam& param_ = nnvm::get<NormConvolutionParam>(attrs.parsed);
  if (!param_.no_norm) {
    CHECK_EQ(in_shape->size(), static_cast<size_t>(norm_conv::kNumInputs))
      << "Input:[data, in_sum, in_sum_squares, gamma, beta, moving_mean, moving_var, weight]";
  } else {
    CHECK_EQ(in_shape->size(), static_cast<size_t>(norm_conv::kNumInputsNoNorm))
      << "Input:[data, weight]";
  }
  int num_outputs = norm_conv::NumOutputs(param_.no_norm);
  CHECK_EQ(out_shape->size(), static_cast<size_t>(num_outputs));
  int weight_idx = norm_conv::WeightIdx(param_.no_norm);
  out_shape->resize(num_outputs, TShape());
  const TShape &dshp = (*in_shape)[norm_conv::kData];
  if (dshp.ndim() ==  0) return false;

  SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kOutSum, Shape1(param_.num_filter));
  SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kOutSumOfSquares, Shape1(param_.num_filter));

  if (param_.kernel.ndim() == 1) {
    // 1d conv
    CHECK_EQ(dshp.ndim(), 3U) << "Input data should be 3D in batch-num_filter-x";
    Shape<3> dshape = ConvertLayout(dshp.get<3>(), param_.layout.value(), kNCW);
    Shape<3> wshape = Shape3(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel[0]);
    wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, weight_idx, wshape);
    if (!param_.no_norm) {
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kInSum, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kInSumOfSquares, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kGamma, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kBeta, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kMovingMean, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kMovingVar, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kSavedMean, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kSavedInvStdDev, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kEquivScale, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kEquivBias, Shape1(dshape[1]));
    }

    const index_t dilated_ksize_x = param_.DilatedKernelSize(0);
    CHECK_EQ(dshape[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<3> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_x) / param_.stride[0] + 1 : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kOut,
                       ConvertLayout(oshape, kNCW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[norm_conv::kOut].get<3>(), param_.layout.value(), kNCW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_x - 1 - 2 * param_.pad[0];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kData,
        ConvertLayout(dshape, kNCW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    return true;
  } else if (param_.kernel.ndim() == 2) {
    // 2d conv
    CHECK_EQ(dshp.ndim(), 4U) \
      << "Input data should be 4D in batch-num_filter-y-x";
    Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
    Shape<4> wshape = Shape4(param_.num_filter / param_.num_group,
        dshape[1] / param_.num_group,
        param_.kernel[0], param_.kernel[1]);
    wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, weight_idx, wshape);
    if (!param_.no_norm) {
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kInSum, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kInSumOfSquares, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kGamma, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kBeta, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kMovingMean, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kMovingVar, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kSavedMean, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kSavedInvStdDev, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kEquivScale, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kEquivBias, Shape1(dshape[1]));
    }

    const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
    const index_t dilated_ksize_x = param_.DilatedKernelSize(1);
    CHECK_EQ(dshape[1] % param_.num_group, 0U) \
      << "input num_filter must divide group size";
    CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
      << "output num_filter must divide group size";
    CHECK_GT(param_.kernel.Size(), 0U) \
      << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
      << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
      << "incorrect dilate size: " << param_.dilate;
    Shape<4> oshape;
    oshape[0] = dshape[0];
    oshape[1] = param_.num_filter;
    oshape[2] = dshape[2] ?
      (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_y) / param_.stride[0] + 1 : 0;
    oshape[3] = dshape[3] ?
      (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_x) / param_.stride[1] + 1 : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, norm_conv::kOut,
                       ConvertLayout(oshape, kNCHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[norm_conv::kOut].get<4>(), param_.layout.value(), kNCHW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] && param_.stride[1] == 1) {
      dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, norm_conv::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
      CHECK_LE(dilated_ksize_y, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
    }
    if (dshape[3] != 0) {
      CHECK_LE(dilated_ksize_x, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
    }
    return true;
  } else if (param_.kernel.ndim() == 3) {
    LOG(FATAL) << "3D NormConvolution not supported.";
    return true;
  } else {
    LOG(FATAL) << "Unknown normConvolution type";
    return false;
  }
}

static bool NormConvolutionType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type, std::vector<int> *out_type) {
  const NormConvolutionParam& param_ = nnvm::get<NormConvolutionParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // For float16 input type, equiv_scale, equiv_bias, weights and output are also float16,
  // but moving_mean and moving_var inputs, and out_sum and out_sum_of_squares outputs are float32.

  int dtype_statistics;

  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_statistics = mshadow::DataType<AccRealX>::kFlag; });

  // Default expected input dtype matches that of the 1st (i.e. data) input.
  std::vector<int> input_types(in_type->size(), dtype);
  if (!param_.no_norm) {
    // in_sum, in_sum_squares, moving_mean, moving_var, gamma and beta have increased precision.
    input_types[norm_conv::kInSum] = dtype_statistics;
    input_types[norm_conv::kInSumOfSquares] = dtype_statistics;
    input_types[norm_conv::kGamma] = dtype_statistics;
    input_types[norm_conv::kBeta] = dtype_statistics;
    input_types[norm_conv::kMovingMean] = dtype_statistics;
    input_types[norm_conv::kMovingVar] = dtype_statistics;
  }
  for (size_t i = 0; i < in_type->size(); ++i) {
    int expected_type = input_types[i];
    if ((*in_type)[i] == -1) {
      (*in_type)[i] = expected_type;
    } else {
      UNIFORM_TYPE_CHECK((*in_type)[i], expected_type, ListArguments(param_)[i]);
    }
  }
  out_type->clear();
  for (int i = 0; i != norm_conv::NumOutputs(param_.no_norm); ++i) {
    if (i == norm_conv::kOut || i == norm_conv::kEquivScale || i == norm_conv::kEquivBias)
      out_type->push_back(dtype);
    else
      out_type->push_back(dtype_statistics);
  }
  return true;
}


void NormConvolutionParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  NormConvolutionParam param_;
  try {
    param_.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }

  if (param_.kernel.ndim() == 1) {
    param_.layout = param_.layout? param_.layout.value() : mshadow::kNCW;
    if (param_.stride.ndim() == 0) param_.stride = Shape1(1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape1(1);
    if (param_.pad.ndim() == 0) param_.pad = Shape1(0);
  } else if (param_.kernel.ndim() == 2) {
    param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
    if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
  } else {
    CHECK_EQ(param_.kernel.ndim(), 3U) << param_.kernel.ndim()
                                       << "3D NormConvolution not supported";
    param_.layout = param_.layout ? param_.layout.value(): mshadow::kNCDHW;
    if (param_.stride.ndim() == 0) param_.stride = Shape3(1, 1, 1);
    if (param_.dilate.ndim() == 0) param_.dilate = Shape3(1, 1, 1);
    if (param_.pad.ndim() == 0) param_.pad = Shape3(0, 0, 0);
  }
  CHECK_EQ(param_.kernel.ndim(), param_.stride.ndim())
    << "Stride must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while stride is "
    << param_.stride;
  CHECK_EQ(param_.kernel.ndim(), param_.dilate.ndim())
    << "Dilate must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while dilate is "
    << param_.dilate;
  CHECK_EQ(param_.kernel.ndim(), param_.pad.ndim())
    << "Padding must have the same number of dimensions with kernel_size,"
    << "but kernel_size is set to " << param_.kernel << " while padding is "
    << param_.pad;
  attrs->parsed = std::move(param_);
}

struct NormConvolutionGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const NormConvolutionParam& param_ = nnvm::get<NormConvolutionParam>(n->attrs.parsed);
    size_t num_fwd_inputs = n->inputs.size();
    size_t num_fwd_outputs = n->num_outputs();
    if (!param_.no_norm) {
      CHECK_EQ(num_fwd_inputs, static_cast<size_t>(norm_conv::kNumInputs))
        << "Input:[data, in_sum, in_sum_squares, moving_mean, moving_var, gamma, beta, weight]";
    } else {
      CHECK_EQ(num_fwd_inputs, static_cast<size_t>(norm_conv::kNumInputsNoNorm))
        << "Input:[data, weight]";
    }

    // We compose the inputs to the backward node from select 'ograds' gradients, fwd node inputs
    // and fwd node outputs:
    //
    // We take only the first output gradient (d_out).
    // We omit the first 3 outputs (out, sum_out and sum_squares_out), and take the remainder of the
    //   outputs (saved_mean, saved_inv_stddev, equiv_scale, equiv_bias) if (no_norm == false).
    // For simplicity of index calculation, we take all inputs.

    std::vector<nnvm::NodeEntry> heads;

    heads.reserve(1 + (num_fwd_outputs-3) + num_fwd_inputs);
    CHECK_GT(ograds.size(), norm_conv::kData)
      << "Not enough gradients of NormConvolution node.";
    // Copy only the output gradient of the primary output to the backward node
    heads.push_back(ograds[norm_conv::kOut]);

    // Copy all but 3 outputs of forward node to backward node
    for (uint32_t i = 0; i < num_fwd_outputs; ++i) {
      if (i != norm_conv::kOut && i != norm_conv::kOutSum && i != norm_conv::kOutSumOfSquares)
        heads.push_back(nnvm::NodeEntry{n, i, 0});
    }

    // Copy all inputs of forward node to backward node
    for (uint32_t i = 0; i < num_fwd_inputs; ++i) {
      heads.push_back(n->inputs[i]);
    }

    nnvm::NodePtr gnode = nnvm::Node::Create();
    gnode->inputs = std::move(heads);
    gnode->control_deps.emplace_back(n);
    gnode->attrs = n->attrs;
    gnode->attrs.op = nnvm::Op::Get("_backward_NormConvolution");
    gnode->attrs.name = n->attrs.name + "_backward";

    // Prepare a no-gradient node to forbid gradient on aux_states
    nnvm::NodePtr ng = nnvm::Node::Create();
    ng->attrs.op = Op::Get("_NoGradient");
    ng->attrs.name = "NoGradient";

    // The set the vector of input gradients
    std::vector<nnvm::NodeEntry> in_grad(num_fwd_inputs);
    uint32_t from_index = 0;
    for (uint32_t i = 0; i < num_fwd_inputs; ++i) {
      if (i == norm_conv::kMovingMean || i == norm_conv::kMovingVar)
        in_grad[i] = nnvm::NodeEntry{ng, 0, 0};
      else
        in_grad[i] = nnvm::NodeEntry{gnode, from_index++, 0};
    }
    return in_grad;
  }
};

NNVM_REGISTER_OP(NormConvolution)
.describe(R"code(Compute *N*-D normConvolution on *(N+2)*-D input.

******** Documentation not yet correct for this fused normalized convolution!! *************

In the 2-D normConvolution, given input data with shape *(batch_size,
channel, height, width)*, the output is computed by

.. math::

   out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
   weight[i,j,:,:]

where :math:`\star` is the 2-D cross-correlation operator.

For general 2-D normConvolution, the shapes are

- **data**: *(batch_size, channel, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_height, out_width)*.

Define::

  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

then we have::

  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
width)*. We can choose other layouts such as *NWC*.

If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
evenly into *g* parts along the channel axis, and also evenly split ``weight``
along the first dimension. Next compute the normConvolution on the *i*-th part of
the data with the *i*-th weight part. The output is obtained by concatenating all
the *g* results.

1-D normConvolution does not have *height* dimension but only *width* in space.

- **data**: *(batch_size, channel, width)*
- **weight**: *(num_filter, channel, kernel[0])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_width)*.

3-D normConvolution adds an additional *depth* dimension besides *height* and
*width*. The shapes are

- **data**: *(batch_size, channel, depth, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.

Both ``weight`` and ``bias`` are learnable parameters.

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  return norm_conv::NumInputs(params.no_norm);
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  return norm_conv::NumOutputs(params.no_norm);
})
.set_attr_parser(NormConvolutionParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  if (params.no_norm)
    return std::vector<std::string>{"data", "weight"};
  else
    return std::vector<std::string>{"data", "in_sum", "in_sum_squares", "gamma", "beta",
                                    "moving_mean", "moving_var", "weight"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "out_sum", "out_sum_squares",
                                  "saved_mean", "saved_inv_stddev", "equiv_scale", "equiv_bias"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NormConvolutionShape)
.set_attr<nnvm::FInferType>("FInferType", NormConvolutionType)
.set_attr<FCompute>("FCompute<cpu>", NormConvolutionCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                           NormConvolutionGrad{"_backward_NormConvolution"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  uint32_t visible_outputs = norm_conv::NumOutputs(params.no_norm);
  if (!params.no_norm && !params.output_equiv_scale_bias) {
    visible_outputs -= 2;
    // If equiv_scale and equiv_bias are Python-visible (for testing),
    // then mean and var must be also visible.
    if (!params.output_mean_var)
      visible_outputs -= 2;
  }
  return visible_outputs;
})
.set_attr<nnvm::FMutateInputs>("FMutateInputs", [](const nnvm::NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  if (params.no_norm)
    return std::vector<uint32_t>{};
  else
    return std::vector<uint32_t>{norm_conv::kMovingMean, norm_conv::kMovingVar};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to the NormConvolutionOp.")
.add_argument("sum", "NDArray-or-Symbol", "sum of input data to be normalized")
.add_argument("sum_squares", "NDArray-or-Symbol", "sum of squares of input data to be normalized")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("moving_mean", "NDArray-or-Symbol", "running mean of input")
.add_argument("moving_var", "NDArray-or-Symbol", "running variance of input")
.add_arguments(NormConvolutionParam::__FIELDS__())
.set_attr<nnvm::FSetInputVarAttrOnCompose>(
  "FSetInputVarAttrOnCompose",
  [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
    if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
    if (index == norm_conv::kMovingMean) {
      var->attrs.dict["__init__"] = "[\"zero\", {}]";
    } else if (index == norm_conv::kMovingVar) {
      var->attrs.dict["__init__"] = "[\"one\", {}]";
    }
  });

NNVM_REGISTER_OP(_backward_NormConvolution)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  size_t num_fwd_inputs = norm_conv::NumInputs(params.no_norm);
  size_t num_fwd_outputs = norm_conv::NumOutputs(params.no_norm);
  return 1 + (num_fwd_outputs-3) + num_fwd_inputs;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const NormConvolutionParam& params = nnvm::get<NormConvolutionParam>(attrs.parsed);
  // The outputs of the backward node are the fwd-node input gradients, so one per fwd node input.
  return norm_conv::NumNonAuxInputs(params.no_norm);
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(NormConvolutionParamParser)
.set_attr<FCompute>("FCompute<cpu>", NormConvolutionGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
