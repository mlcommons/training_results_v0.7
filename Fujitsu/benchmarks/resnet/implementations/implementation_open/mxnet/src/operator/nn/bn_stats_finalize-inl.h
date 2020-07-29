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
#ifndef MXNET_OPERATOR_NN_BN_STATS_FINALIZE_INL_H_
#define MXNET_OPERATOR_NN_BN_STATS_FINALIZE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mshadow/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"
#include "../mxnet_op.h"

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

namespace mxnet {
namespace op {

namespace bn_stats_finalize {
  enum BatchNormOpInputs {kSum, kSumOfSquares, kGamma, kBeta, kInMovingMean, kInMovingVar,
    kNumInputs  // Not an I/O! Leave this at the end
  };  // kGamma: weights, kBeta: biases
  enum BatchNormOpOutputs {kEquivScale, kEquivBias, kMean, kInvStdDev, kGammaOut, kBetaOut,
    kNumOutputs  // Not an I/O! Leave this at the end
  };  // req, out_data
  enum BatchNormOpAuxiliary {kMovingMean, kMovingVar,
    kNumAuxStates  // Not an I/O! Leave this at the end
  };  // aux_states
  enum BatchNormCounts {kNumNonAuxInputs = (kNumInputs - kNumAuxStates)};
}  // namespace bn_stats_finalize

/*! \brief Parameters for BatchNoram operator */
struct BNStatsFinalizeParam : public dmlc::Parameter<BNStatsFinalizeParam> {
  double eps;
  float momentum;
  bool fix_gamma;
  bool use_global_stats;
  bool output_mean_var;
  uint64_t elem_count;
  DMLC_DECLARE_PARAMETER(BNStatsFinalizeParam) {
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
    .describe("Output the mean and inverse std ");
    DMLC_DECLARE_FIELD(elem_count)
    .describe("Number of elements accumulated in 'sum' and 'sum_squares'.");
  }

  bool operator==(const BNStatsFinalizeParam& other) const {
    return this->eps == other.eps &&
           this->momentum == other.momentum &&
           this->fix_gamma == other.fix_gamma &&
           this->use_global_stats == other.use_global_stats &&
           this->output_mean_var == other.output_mean_var &&
           this->elem_count == other.elem_count;
  }
};

typedef ParamOpSign<BNStatsFinalizeParam> BNStatsFinalizeSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::BNStatsFinalizeParam> {
  size_t operator()(const mxnet::op::BNStatsFinalizeParam& val) {
    size_t ret = 0;
    // Not critical, but not sure why eps is left out.  Related to being 'double'?
    ret = dmlc::HashCombine(ret, val.momentum);
    ret = dmlc::HashCombine(ret, val.fix_gamma);
    ret = dmlc::HashCombine(ret, val.use_global_stats);
    ret = dmlc::HashCombine(ret, val.output_mean_var);
    ret = dmlc::HashCombine(ret, val.elem_count);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

template<typename xpu>
void BNStatsFinalizeCompute(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx, const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "Only gpu impl of BNStatsFinalize exists.";
}

template<typename xpu>
void BNStatsFinalizeGradCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "Only gpu impl of BNStatsFinalize exists.";
}

}  // namespace op
}  // namespace mxnet

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

#endif  // MXNET_OPERATOR_NN_BN_STATS_FINALIZE_INL_H_

