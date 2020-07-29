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
 * \file norm_convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \author Dick Carter
*/
#ifndef MXNET_OPERATOR_NN_NORM_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_NORM_CONVOLUTION_INL_H_

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

namespace norm_conv {
  // Since the optional inputs are not at the end, the weight index is parameter dependent.
  enum NormConvolutionOpInputs {kData, kInSum, kInSumOfSquares,
                                kGamma, kBeta, kMovingMean, kMovingVar, kWeight,
    kNumInputs  // Not an I/O! Leave this at the end
  };
  enum NormConvolutionOpInputsNoNorm {kDataNoNorm, kWeightNoNorm,
    kNumInputsNoNorm  // Not an I/O! Leave this at the end
  };
  enum NormConvolutionOpOutputs {kOut, kOutSum, kOutSumOfSquares, kSavedMean, kSavedInvStdDev,
                                 kEquivScale, kEquivBias,
    kNumOutputs  // Not an I/O! Leave this at the end
  };
  enum NormConvolutionOpOutputsNoNorm {kOutNoNorm, kOutSumNoNorm, kOutSumOfSquaresNoNorm,
    kNumOutputsNoNorm  // Not an I/O! Leave this at the end
  };
  enum NormConvolutionOpAuxiliary {kAuxMovingMean, kAuxMovingVar,
    kNumAuxStates  // Not an I/O! Leave this at the end
  };  // aux_states
  // The kOut, kOutSum and kOutSumOfSquares of the forward node are not used in the backward node.
  enum BwdNormConvolutionOpOutputs {kBwdSavedMean, kBwdSavedInvStdDev,
                                    kBwdEquivScale, kBwdEquivBias,
    kBwdNumUsedFwdOutputs  // Not an I/O! Leave this at the end
  };
  enum NormConvolutionOpResource {kTempSpace};
  inline int WeightIdx(bool no_norm) {
    return no_norm ? static_cast<int>(kWeightNoNorm) : static_cast<int>(kWeight);
  }
  // Adjusts for aux_inputs not having gradients.  Expected usages:
  //     BwdInGradIdx(kGamma)
  //     BwdInGradIdx(kBeta)
  //     BwdInGradIdx(Weightidx(no_norm))
  inline int BwdInGradIdx(int fwd_input_idx) {
    return (fwd_input_idx > kMovingVar) ? static_cast<int>(fwd_input_idx - kNumAuxStates)
                                        : fwd_input_idx;
  }
  inline int NumInputs(bool no_norm) {
    return no_norm ? static_cast<int>(kNumInputsNoNorm) : static_cast<int>(kNumInputs);
  }
  inline int NumOutputs(bool no_norm) {
    return no_norm ? static_cast<int>(kNumOutputsNoNorm) : static_cast<int>(kNumOutputs);
  }
  inline int NumNonAuxInputs(bool no_norm) {
    return no_norm ? static_cast<int>(kNumInputsNoNorm)
                   : static_cast<int>(kNumInputs - kNumAuxStates);
  }
  bool IsCompatibleConvolution(const nnvm::NodePtr& node, const int& dtype,
                               const TShape& shape, const Context& ctx);
}  // namespace norm_conv

struct NormConvolutionParam : public dmlc::Parameter<NormConvolutionParam> {
  mxnet::TShape kernel;
  mxnet::TShape stride;
  mxnet::TShape dilate;
  mxnet::TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  bool no_norm;
  dmlc::optional<int> layout;
  dmlc::optional<int> act_type;
  // Parameters inherited from 'Finalize'
  double eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  bool output_equiv_scale_bias;
  DMLC_DECLARE_PARAMETER(NormConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel)
    .describe("NormConvolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(mxnet::TShape(0, 0))
    .describe("NormConvolution stride: (w,), (h, w) or (d, h, w). Default 1 for each dim.");
    DMLC_DECLARE_FIELD(dilate).set_default(mxnet::TShape(0, 0))
    .describe("NormConvolution dilate: (w,), (h, w) or (d, h, w). Default 1 for each dim.");
    DMLC_DECLARE_FIELD(pad).set_default(mxnet::TShape(0, 0))
    .describe("Zero pad for NormConvolution: (w,), (h, w) or (d, h, w). Default no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("NormConvolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(no_norm).set_default(false)
    .describe("Whether to disable input normalization prior to the convolution.");
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
    // More fields from 'Finalize'
    DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
    .describe("Epsilon to prevent div 0. "
              "Must be no less than CUDNN_BN_MIN_EPSILON "
              "defined in cudnn.h when using cudnn (usually 1e-5)");
    DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
    .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Fix gamma while training");
    DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
    .describe("Whether use global moving statistics instead of local batch-norm. "
              "This will force change batch-norm into a scale shift operator.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
    .describe("Output the mean and inverse std.");
    DMLC_DECLARE_FIELD(output_equiv_scale_bias).set_default(false)
    .describe("Output the equiv_scale and equiv_bias (generally for testing).");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const NormConvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->no_norm == other.no_norm &&
           this->layout == other.layout &&
           this->act_type == other.act_type &&
           this->eps == other.eps &&
           this->momentum == other.momentum &&
           this->fix_gamma == other.fix_gamma &&
           this->use_global_stats == other.use_global_stats &&
           this->output_mean_var == other.output_mean_var &&
           this->output_equiv_scale_bias == other.output_equiv_scale_bias;
  }
};

void NormConvolutionParamParser(nnvm::NodeAttrs* attrs);

typedef ParamOpSign<NormConvolutionParam> NormConvSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::NormConvolutionParam> {
  size_t operator()(const mxnet::op::NormConvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.num_filter);
    ret = dmlc::HashCombine(ret, val.num_group);
    ret = dmlc::HashCombine(ret, val.no_norm);
    ret = dmlc::HashCombine(ret, val.layout);
    ret = dmlc::HashCombine(ret, val.act_type ? val.act_type.value() : -1);
    ret = dmlc::HashCombine(ret, val.momentum);
    ret = dmlc::HashCombine(ret, val.fix_gamma);
    ret = dmlc::HashCombine(ret, val.use_global_stats);
    ret = dmlc::HashCombine(ret, val.output_mean_var);
    ret = dmlc::HashCombine(ret, val.output_equiv_scale_bias);

    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu, typename DType>
class NormConvolutionOp {
 public:
  void Init(NormConvolutionParam p) {
    LOG(FATAL) << "Only cudnn-based NormConvolution supported.";
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
    LOG(FATAL) << "Only cudnn-based NormConvolution supported.";
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    LOG(FATAL) << "Only cudnn-based NormConvolution supported.";
  }

 private:
  NormConvolutionParam param_;
};  // class NormConvolutionOp

template<typename xpu>
void NormConvolutionCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx, const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const NormConvolutionParam& param = nnvm::get<NormConvolutionParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[norm_conv::kData].type_flag_, DType, {
    NormConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void NormConvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const NormConvolutionParam& param = nnvm::get<NormConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    NormConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_NORM_CONVOLUTION_INL_H_
