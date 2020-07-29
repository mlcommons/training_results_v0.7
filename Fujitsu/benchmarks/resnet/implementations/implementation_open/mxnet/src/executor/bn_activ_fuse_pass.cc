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
 * \file bn_activation_fuse_pass.cc
 * \brief optimization pass which fuse Activation into BatchNorm
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>

#include "./exec_pass.h"
#include "../operator/nn/activation-inl.h"
#include "../operator/nn/batch_norm-inl.h"

namespace mxnet {
namespace exec {

using namespace mxnet::op;

namespace {
  inline bool IsCompatibleActivation(const nnvm::NodePtr& node) {
    if (node->op() == Op::Get("Activation")) {
      const auto act_type = nnvm::get<ActivationParam>(node->attrs.parsed).act_type;
      return act_type == activation::kReLU;
    }

    if (node->op() == Op::Get("relu")) {
      node->attrs.dict["act_type"] = "relu";
    }
    return node->op() == Op::Get("relu");
  }

  inline bool IsCompatibleBN(const nnvm::NodePtr& node, const int& dtype,
                             const TShape& shape, const Context& ctx) {
    if (node->op() != Op::Get("BatchNorm"))
      return false;
    return batchnorm::SupportsFusedActivation(node, dtype, shape, ctx);
  }
}  // namespace

Graph FuseBNActiv(Graph&& g) {
  const auto& shape_vec   = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtype_vec   = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& context_vec = g.GetAttr<ContextVector>("context");
  const auto& ig = g.indexed_graph();
  const auto ne_counter = GetNodeEntryCount(g);
  nnvm::NodeEntryMap<nnvm::NodeEntry> entry_map;
  DFSVisit(g.outputs, [&ne_counter, &entry_map, &ig, &shape_vec, &dtype_vec,
                       &context_vec](const nnvm::NodePtr& n) {
    if (IsCompatibleActivation(n)) {
      auto bn = n->inputs[0].node;
      auto nid = ig.node_id(bn.get());
      auto eid = ig.entry_id(nid, 0);
      if (IsCompatibleBN(bn, dtype_vec[eid], shape_vec[eid], context_vec[nid]) &&
          n->inputs[0].index == 0 &&
          ne_counter.at(n->inputs[0]) == 1) {
        bn->attrs.dict["act_type"] = n->attrs.dict["act_type"];
        bn->attrs.name += "_activ";
        bn->op()->attr_parser(&(bn->attrs));
        entry_map.insert({nnvm::NodeEntry{n, 0, 0}, nnvm::NodeEntry{bn, 0, 0}});
      }
    }
  });
  g = ReplaceNodeEntries(std::move(g), entry_map);
  return g;
}

}  // namespace exec
}  // namespace mxnet
