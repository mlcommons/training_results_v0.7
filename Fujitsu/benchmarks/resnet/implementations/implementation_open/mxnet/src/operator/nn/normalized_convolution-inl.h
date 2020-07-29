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
 * \file normalized_convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \author Dick Carter
*/
#ifndef MXNET_OPERATOR_NN_NORMALIZED_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_NORMALIZED_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
#include "./im2col.h"
#include "activation-inl.h"

namespace mxnet {
namespace op {

namespace normalized_conv {
  // Since the optional inputs are not at the end, the weight index is parameter dependent.
  enum NormalizedConvolutionOpInputs {kData, kEquivScale, kEquivBias,
                                      kMean, kVar, kGamma, kBeta, kWeight,
    kNumInputs  // Not an I/O! Leave this at the end
  };
  enum NormalizedConvolutionOpInputsNoEquivScaleBias {kDataNoEquivScaleBias,
                                                      kWeightNoEquivScaleBias,
    kNumInputsNoEquivScaleBias  // Not an I/O! Leave this at the end
  };
  enum NormalizedConvolutionOpOutputs {kOut, kSum, kSumOfSquares,
    kNumOutputs  // Not an I/O! Leave this at the end
  };
  enum NormalizedConvolutionOpResource {kTempSpace};
  inline int WeightIdx(bool no_equiv_scale_bias) {
    return no_equiv_scale_bias ? static_cast<int>(kWeightNoEquivScaleBias)
                               : static_cast<int>(kWeight);
  }
  inline int NumInputs(bool no_equiv_scale_bias) {
    return no_equiv_scale_bias ? static_cast<int>(kNumInputsNoEquivScaleBias)
                               : static_cast<int>(kNumInputs);
  }
}  // namespace normalized_conv

struct NormalizedConvolutionParam : public dmlc::Parameter<NormalizedConvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  bool no_equiv_scale_bias;
  dmlc::optional<int> layout;
  dmlc::optional<int> act_type;
  DMLC_DECLARE_PARAMETER(NormalizedConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel)
    .describe("NormalizedConvolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
    .describe("NormalizedConvolution stride: (w,), (h, w) or (d, h, w). Default 1 for each dim.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
    .describe("NormalizedConvolution dilate: (w,), (h, w) or (d, h, w). Default 1 for each dim.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
    .describe("Zero pad for NormalizedConvolution: (w,), (h, w) or (d, h, w). Default no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("NormalizedConvolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(no_equiv_scale_bias).set_default(false)
    .describe("Whether to disable normalization equivalent-scale and equivalent-bias adjustments.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d."
              "NHWC and NDHWC are only supported on GPU.");
    // Should track the definition in activation-inl.h
    DMLC_DECLARE_FIELD(act_type)
    .add_enum("relu", activation::kReLU)
    .add_enum("sigmoid", activation::kSigmoid)
    .add_enum("tanh", activation::kTanh)
    .add_enum("softrelu", activation::kSoftReLU)
    .set_default(dmlc::optional<int>())
    .describe("Fused activation function to be applied.");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const NormalizedConvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->no_equiv_scale_bias == other.no_equiv_scale_bias &&
           this->layout == other.layout &&
           this->act_type == other.act_type;
  }
};

void NormalizedConvolutionParamParser(nnvm::NodeAttrs* attrs);

typedef ParamOpSign<NormalizedConvolutionParam> NormalizedConvSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::NormalizedConvolutionParam> {
  size_t operator()(const mxnet::op::NormalizedConvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.num_filter);
    ret = dmlc::HashCombine(ret, val.num_group);
    ret = dmlc::HashCombine(ret, val.no_equiv_scale_bias);
    ret = dmlc::HashCombine(ret, val.layout);
    ret = dmlc::HashCombine(ret, val.act_type ? val.act_type.value() : -1);

    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class NormalizedConvolutionOp {
 public:
  void Init(NormalizedConvolutionParam p) {
    LOG(FATAL) << "Only cudnn-based NormalizedConvolution supported.";
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    CHECK(param_.layout.value() == mshadow::kNCW ||
          param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    LOG(FATAL) << "Only cudnn-based NormalizedConvolution supported.";
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    LOG(FATAL) << "Only cudnn-based NormalizedConvolution supported.";
  }

 private:
  NormalizedConvolutionParam param_;
};  // class NormalizedConvolutionOp

template<typename xpu>
void NormalizedConvolutionCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const NormalizedConvolutionParam& param = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[normalized_conv::kData].type_flag_, DType, {
    NormalizedConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void NormalizedConvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const NormalizedConvolutionParam& param = nnvm::get<NormalizedConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    NormalizedConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_NORMALIZED_CONVOLUTION_INL_H_
