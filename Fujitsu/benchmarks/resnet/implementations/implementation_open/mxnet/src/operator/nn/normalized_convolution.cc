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
 * \file normalized_convolution.cc
 * \brief
 * \author Dick Carter
*/

#include "./normalized_convolution-inl.h"
#include "../elemwise_op_common.h"
#include "../operator_common.h"
#if MXNET_USE_NNPACK == 1
#include "../nnpack/nnpack_pooling-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(NormalizedConvolutionParam);

static inline index_t AddPad(index_t dsize, index_t pad) {
  return dsize + 2 * pad;
}

static inline std::vector<std::string> ListArguments(const NormalizedConvolutionParam& param_) {
  if (!param_.no_equiv_scale_bias) {
    return {"data", "equiv_scale", "equiv_bias", "mean", "var", "weight"};
  } else {
    return {"data", "weight"};
  }
}

static bool NormalizedConvolutionShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_shape,
                             std::vector<TShape> *out_shape) {
  using namespace mshadow;
  const NormalizedConvolutionParam& param_ = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  if (!param_.no_equiv_scale_bias) {
    CHECK_EQ(in_shape->size(), static_cast<size_t>(normalized_conv::kNumInputs))
      << "Input:[data, equiv_scale, equiv_bias, mean, var, gamma, beta, weight]";
  } else {
    CHECK_EQ(in_shape->size(), static_cast<size_t>(normalized_conv::kNumInputsNoEquivScaleBias))
      << "Input:[data, weight]";
  }
  int weight_idx = normalized_conv::WeightIdx(param_.no_equiv_scale_bias);
  out_shape->resize(normalized_conv::kNumOutputs, TShape());
  const TShape &dshp = (*in_shape)[normalized_conv::kData];
  if (dshp.ndim() ==  0) return false;

  SHAPE_ASSIGN_CHECK(*out_shape, normalized_conv::kSum, Shape1(param_.num_filter));
  SHAPE_ASSIGN_CHECK(*out_shape, normalized_conv::kSumOfSquares, Shape1(param_.num_filter));

  if (param_.kernel.ndim() == 1) {
    // 1d conv
    CHECK_EQ(dshp.ndim(), 3U) << "Input data should be 3D in batch-num_filter-x";
    Shape<3> dshape = ConvertLayout(dshp.get<3>(), param_.layout.value(), kNCW);
    Shape<3> wshape = Shape3(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel[0]);
    wshape = ConvertLayout(wshape, kNCW, param_.layout.value());
    wshape[0] *= param_.num_group;
    SHAPE_ASSIGN_CHECK(*in_shape, weight_idx, wshape);
    if (!param_.no_equiv_scale_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kEquivScale, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kEquivBias, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kMean, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kVar, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kGamma, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kBeta, Shape1(dshape[1]));
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
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<3>(), param_.layout.value(), kNCW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_x - 1 - 2 * param_.pad[0];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kData,
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
    if (!param_.no_equiv_scale_bias) {
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kEquivScale, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kEquivBias, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kMean, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kVar, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kGamma, Shape1(dshape[1]));
      SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kBeta, Shape1(dshape[1]));
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
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
      dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] && param_.stride[1] == 1) {
      dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, normalized_conv::kData,
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
    LOG(FATAL) << "3D NormalizedConvolution not supported.";
    return true;
  } else {
    LOG(FATAL) << "Unknown normalizedConvolution type";
    return false;
  }
}

static bool NormalizedConvolutionType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_type, std::vector<int> *out_type) {
  const NormalizedConvolutionParam& param_ = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  CHECK_GE(in_type->size(), 1U);
  int dtype = (*in_type)[0];
  CHECK_NE(dtype, -1) << "First input must have specified type";
  // For float16 input type, the equiv_scale, equiv_bias, weights and output are also float16,
  // but the mean and var inputs, and the sum and sum_of_squares outputs are stored in float32.

  int dtype_statistics;

  MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DTypeX, AccRealX, {
      dtype_statistics = mshadow::DataType<AccRealX>::kFlag; });

  // Default expected input dtype matches that of the 1st (i.e. data) input.
  std::vector<int> input_types(in_type->size(), dtype);
  // However the 'mean', 'var', 'gamma' and 'beta' inputs, if present, may have increased precision.
  if (!param_.no_equiv_scale_bias) {
    input_types[normalized_conv::kMean] = dtype_statistics;
    input_types[normalized_conv::kVar] = dtype_statistics;
    input_types[normalized_conv::kGamma] = dtype_statistics;
    input_types[normalized_conv::kBeta] = dtype_statistics;
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
  // 1st data output is of type 'dtype', rest are of greater precision 'dtype_statistics'
  out_type->push_back(dtype);
  while (out_type->size() < static_cast<size_t>(normalized_conv::kNumOutputs))
    out_type->push_back(dtype_statistics);
  return true;
}


void NormalizedConvolutionParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;
  NormalizedConvolutionParam param_;
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
                                       << "3D NormalizedConvolution not supported";
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

struct NormalizedConvolutionGrad {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) const {
    const NormalizedConvolutionParam& param_ =
        nnvm::get<NormalizedConvolutionParam>(n->attrs.parsed);
    size_t num_fwd_inputs = n->inputs.size();
    size_t num_fwd_outputs = n->num_outputs();
    if (!param_.no_equiv_scale_bias) {
      CHECK_EQ(num_fwd_inputs, static_cast<size_t>(normalized_conv::kNumInputs))
        << "Input:[data, equiv_scale, equiv_bias, mean, var, gamma, beta, weight]";
    } else {
      CHECK_EQ(num_fwd_inputs, static_cast<size_t>(normalized_conv::kNumInputsNoEquivScaleBias))
        << "Input:[data, weight]";
    }

    std::vector<nnvm::NodeEntry> heads;
    // We copy the outputs and the inputs of the forward node to the inputs of the backward node,
    // with the one *important* exception that the first backward input is the gradient of the first
    // output, not the output itself.  The benefit is that vectors of the forward node output-
    // and input-shapes are easily obtained, as is useful for operator instance lookup and init.

    std::vector<nnvm::NodeEntry> out_data(num_fwd_outputs);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
      out_data[i] = nnvm::NodeEntry{n, i, 0};
    }

    // The one data output gradient, the remainder of the outputs, and all forward node inputs
    // are inputs of the backward node.
    heads.reserve(num_fwd_outputs + num_fwd_inputs);
    CHECK_GT(ograds.size(), normalized_conv::kData)
      << "Not enough gradients of NormalizedConvolution node.";
    // Copy all outputs of forward node to the backward node, but use the gradient of the primary
    // output, instead of the output itself.  Rest are copied to have shape info readily available.
    for (uint32_t i = 0; i < num_fwd_outputs; ++i) {
      heads.push_back((i == normalized_conv::kOut) ? ograds[i] : out_data[i]);
    }
    // Copy all inputs of forward node to backward node
    for (uint32_t i = 0; i < num_fwd_inputs; ++i) {
      heads.push_back(n->inputs[i]);
    }

    nnvm::NodePtr gnode = nnvm::Node::Create();
    gnode->inputs = std::move(heads);
    gnode->control_deps.emplace_back(n);
    gnode->attrs = n->attrs;
    gnode->attrs.op = nnvm::Op::Get("_backward_NormalizedConvolution");
    gnode->attrs.name = n->attrs.name + "_backward";

    std::vector<nnvm::NodeEntry> in_grad(num_fwd_inputs);
    for (uint32_t i = 0; i < num_fwd_inputs; ++i) {
      in_grad[i] = nnvm::NodeEntry{gnode, i, 0};
    }

    return in_grad;
  }
};

NNVM_REGISTER_OP(NormalizedConvolution)
.describe(R"code(Compute *N*-D normalizedConvolution on *(N+2)*-D input.

******** Documentation not yet correct for this fused normalized convolution!! *************

In the 2-D normalizedConvolution, given input data with shape *(batch_size,
channel, height, width)*, the output is computed by

.. math::

   out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
   weight[i,j,:,:]

where :math:`\star` is the 2-D cross-correlation operator.

For general 2-D normalizedConvolution, the shapes are

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
along the first dimension. Next compute the normalizedConvolution on the *i*-th part of
the data with the *i*-th weight part. The output is obtained by concatenating all
the *g* results.

1-D normalizedConvolution does not have *height* dimension but only *width* in space.

- **data**: *(batch_size, channel, width)*
- **weight**: *(num_filter, channel, kernel[0])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_width)*.

3-D normalizedConvolution adds an additional *depth* dimension besides *height* and
*width*. The shapes are

- **data**: *(batch_size, channel, depth, height, width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.

Both ``weight`` and ``bias`` are learnable parameters.

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NormalizedConvolutionParam& params = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  return normalized_conv::NumInputs(params.no_equiv_scale_bias);
})
.set_num_outputs(normalized_conv::kNumOutputs)
.set_attr_parser(NormalizedConvolutionParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  const NormalizedConvolutionParam& params = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  if (params.no_equiv_scale_bias)
    return std::vector<std::string>{"data", "weight"};
  else
    return std::vector<std::string>{"data", "equiv_scale", "equiv_bias", "mean", "var",
                                    "gamma", "beta", "weight"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "sum", "sum_squares"};
})
.set_attr<mxnet::FInferShape>("FInferShape", NormalizedConvolutionShape)
.set_attr<nnvm::FInferType>("FInferType", NormalizedConvolutionType)
.set_attr<FCompute>("FCompute<cpu>", NormalizedConvolutionCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                           NormalizedConvolutionGrad{"_backward_NormalizedConvolution"})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.add_argument("data", "NDArray-or-Symbol", "Input data to the NormalizedConvolutionOp.")
.add_argument("equiv_scale", "NDArray-or-Symbol", "equivalent scale array")
.add_argument("equiv_bias", "NDArray-or-Symbol", "equivalent bias array")
.add_argument("mean", "NDArray-or-Symbol", "mean array")
.add_argument("var", "NDArray-or-Symbol", "array describing variance (actually an inverse std dev)")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array (also known as 'scale')")
.add_argument("beta", "NDArray-or-Symbol", "beta array (also known as 'bias')")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_arguments(NormalizedConvolutionParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_NormalizedConvolution)
.set_num_inputs([](const NodeAttrs& attrs) {
  const NormalizedConvolutionParam& params = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  return normalized_conv::NumInputs(params.no_equiv_scale_bias) + normalized_conv::kNumOutputs;
})
.set_num_outputs([](const NodeAttrs& attrs) {
  const NormalizedConvolutionParam& params = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  // The outputs of the backward node are the fwd-node input gradients, so one per fwd node input.
  return normalized_conv::NumInputs(params.no_equiv_scale_bias);
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr_parser(NormalizedConvolutionParamParser)
.set_attr<FCompute>("FCompute<cpu>", NormalizedConvolutionGradCompute<cpu>);

}  // namespace op
}  // namespace mxnet
