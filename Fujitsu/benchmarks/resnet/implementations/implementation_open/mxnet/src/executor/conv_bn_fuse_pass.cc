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
 * \file convbn_fuse_pass.cc
 * \brief detect whether fused conv + bn is possible
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

#include "./exec_pass.h"
#include "../operator/nn/activation-inl.h"
#include "../operator/nn/batch_norm-inl.h"
#include "../operator/nn/convolution-inl.h"
#include "../operator/nn/norm_convolution-inl.h"
#include "../operator/tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace exec {

using namespace mxnet::op;

namespace {
#if DEBUG
  void PrintGraph(const Graph& g) {
    LOG(INFO) << "######## GRAPH IS ########";
    const auto ne_counter = GetNodeEntryCount(g);
    DFSVisit(g.outputs, [&ne_counter](const nnvm::NodePtr& n) {
      if (n->op() == nullptr) {
        LOG(INFO) << n->attrs.name << ":";
      } else {
        LOG(INFO) << n->attrs.name << ": " << n->op()->name;
      }
      if (n->op() != nullptr) {
        if (n->op() == Op::Get("NormalizedConvolution") ||
            n->op() == Op::Get("Convolution") ||
            n->op() == Op::Get("BatchNorm")) {
          for (const auto& p : n->attrs.dict) {
            LOG(INFO) << "  - " << p.first << ": " << p.second;
          }
        }
        LOG(INFO) << "  INPUTS:";
        for (const auto& e : n->inputs) {
          LOG(INFO) << "  - " << e.node->attrs.name << " | " << e.index
                    << " | " << ne_counter.at(e);
        }
      }
    });
  }
#endif


  bool IsCompatibleBN(const nnvm::NodePtr& node, const TShape& shape) {
    if (node->op() != Op::Get("BatchNorm"))
      return false;
    auto param = nnvm::get<BatchNormParam>(node->attrs.parsed);
    return (shape.ndim() - 1 == param.axis || param.axis == -1);
  }

  void ConvToNormConv(const nnvm::NodePtr& node) {
    static const Op* norm_conv_op = Op::Get("NormConvolution");
    nnvm::NodeAttrs attrs;
    attrs.name = node->attrs.name + "_normalized";
    std::vector<std::string> transferable_conv_dict_params =
        {"kernel", "stride", "dilate", "pad", "num_filter", "num_group", "layout"};
    for (auto& s : transferable_conv_dict_params) {
      const auto& it = node->attrs.dict.find(s);
      if (it != node->attrs.dict.end())
        attrs.dict.insert({s, it->second});
    }
    attrs.dict.insert({"no_norm", "True"});
    attrs.op = norm_conv_op;
    norm_conv_op->attr_parser(&attrs);
    node->attrs = attrs;
  }

  void NormConvToConv(const nnvm::NodePtr& node) {
    static const Op* conv_op = Op::Get("Convolution");
    node->attrs.name.erase(node->attrs.name.end() - 11, node->attrs.name.end());
    node->attrs.op = conv_op;
    node->attrs.dict.erase("no_norm");
    node->attrs.dict.insert({"no_bias", "True"});
    conv_op->attr_parser(&(node->attrs));
  }

  void FuseBatchNorm(const nnvm::NodePtr prev_conv, const nnvm::NodePtr bn,
                     const nnvm::NodePtr next_conv,
                     nnvm::NodeEntryMap<nnvm::NodeEntry>* entry_map) {
    next_conv->attrs.dict["no_norm"] = "False";
    std::vector<std::string> transferable_bn_dict_params =
        {"act_type", "eps", "momentum", "fix_gamma", "use_global_stats", "output_mean_var"};
    for (auto& s : transferable_bn_dict_params) {
      const auto& it = bn->attrs.dict.find(s);
      if (it != bn->attrs.dict.end())
        next_conv->attrs.dict.insert({s, it->second});
    }
    next_conv->inputs.resize(8);
    next_conv->inputs[norm_conv::kData] =
      nnvm::NodeEntry{prev_conv, norm_conv::kOut, 0};
    next_conv->inputs[norm_conv::kWeight] =
      next_conv->inputs[norm_conv::kWeightNoNorm];
    next_conv->inputs[norm_conv::kInSum] =
      nnvm::NodeEntry{prev_conv, norm_conv::kOutSum, 0};
    next_conv->inputs[norm_conv::kInSumOfSquares] =
      nnvm::NodeEntry{prev_conv, norm_conv::kOutSumOfSquares, 0};
    next_conv->inputs[norm_conv::kGamma] =
      bn->inputs[batchnorm::kGamma];
    next_conv->inputs[norm_conv::kBeta] =
      bn->inputs[batchnorm::kBeta];
    next_conv->inputs[norm_conv::kMovingMean] =
      bn->inputs[batchnorm::kInMovingMean];
    next_conv->inputs[norm_conv::kMovingVar] =
      bn->inputs[batchnorm::kInMovingVar];

    next_conv->op()->attr_parser(&(next_conv->attrs));
    entry_map->insert({nnvm::NodeEntry{bn, batchnorm::kMean, 0},
                       nnvm::NodeEntry{next_conv, norm_conv::kSavedMean, 0}});
    entry_map->insert({nnvm::NodeEntry{bn, batchnorm::kVar, 0},
                       nnvm::NodeEntry{next_conv, norm_conv::kSavedInvStdDev, 0}});
  }
}  // namespace

Graph FuseConvBN(Graph&& g) {
  // shapes, dtypes and context are necessary for checking compatibility with NormConvolution
  const auto& shape_vec   = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtype_vec   = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& context_vec = g.GetAttr<ContextVector>("context");
  auto& idx = g.indexed_graph();
  // NormConvolution can have the same behavior than Convolution
  // So we convert compatible Convolution regardless of BN presence
  // We can always convert back to Convolution after
  DFSVisit(g.outputs, [&idx, &shape_vec, &dtype_vec, &context_vec](const nnvm::NodePtr node) {
    if (node->op() == Op::Get("Convolution")) {
      auto eid = idx.entry_id(node->inputs[0]);
      auto nid = idx.node_id(node.get());
      if (norm_conv::IsCompatibleConvolution(node, dtype_vec[eid], shape_vec[eid],
                                             context_vec[nid]))
        ConvToNormConv(node);
    }
  });
  // Fuse NormConv + BN + NormConv => NormConv + NormConv
  auto ne_counter = GetNodeEntryCount(g);
  nnvm::NodeEntryMap<nnvm::NodeEntry> entry_map;
  DFSVisit(g.outputs, [&idx, &shape_vec, &ne_counter, &entry_map](const nnvm::NodePtr& next) {
    if (next->op() == Op::Get("NormConvolution")) {
      auto node = next->inputs[0].node;
      auto eid = idx.entry_id(idx.node_id(node.get()), 0);
      if (IsCompatibleBN(node, shape_vec[eid]) &&
          next->inputs[0].index == 0 &&
          ne_counter[next->inputs[0]] == 1) {
        auto prev = node->inputs[0].node;
        if (prev->op() == Op::Get("NormConvolution") &&
            ne_counter[node->inputs[0]] == 1) {
          FuseBatchNorm(prev, node, next, &entry_map);
        }
      }
    }
  });
  g = ReplaceNodeEntries(std::move(g), entry_map);
  // Transform NormalizedConvolution without any ApplyStats or GenStats to Convolution
  ne_counter = GetNodeEntryCount(g);
  DFSVisit(g.outputs, [&ne_counter](const nnvm::NodePtr& n) {
    if (n->op() == Op::Get("NormConvolution")) {
      auto const& param = nnvm::get<NormConvolutionParam>(n->attrs.parsed);
      if (param.no_norm &&
          ne_counter[nnvm::NodeEntry{n, 1, 0}] == 0 &&
          ne_counter[nnvm::NodeEntry{n, 2, 0}] == 0) {
        NormConvToConv(n);
      }
    }
  });
  return g;
}

}  // namespace exec
}  // namespace mxnet
