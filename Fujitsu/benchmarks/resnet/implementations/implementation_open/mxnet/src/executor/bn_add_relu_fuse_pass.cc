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
 * \file bn_add_relu_fuse_pass.cc
 * \brief optimization pass which fuse add_relu into BatchNorm
 * \author Clement Fuji Tsang
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>

#include "./exec_pass.h"
#include "../operator/nn/batch_norm-inl.h"
#include "../operator/nn/batch_norm_add_relu-inl.h"
#include "../common/cuda_utils.h"

namespace mxnet {
namespace exec {

using namespace mxnet::op;

namespace {

  inline bool IsCompatibleReLU(const nnvm::NodePtr& node) {
    using namespace mxnet::common::cuda;
    if (node->op() == Op::Get("Activation")) {
      const auto& param = nnvm::get<ActivationParam>(node->attrs.parsed);
      return param.act_type == activation::kReLU;
    }

    return node->op() == Op::Get("relu");
  }

  void FuseBNAddReluNode(const nnvm::NodePtr bn, const nnvm::NodePtr add,
                         const nnvm::NodePtr relu, const nnvm::NodeEntry other) {
    static const Op* bn_add_relu_op = Op::Get("BatchNormAddRelu");
    relu->attrs.op = bn_add_relu_op;
    relu->attrs.name = bn->attrs.name + "_add_relu";
    relu->attrs.dict = bn->attrs.dict;
    relu->attrs.dict.erase("act_type");  // BatchNormAddRelu does not have "act_type" parameter
    relu->inputs.resize(6);
    relu->inputs[0] = bn->inputs[0];  // data
    relu->inputs[1] = bn->inputs[1];  // gamma
    relu->inputs[2] = bn->inputs[2];  // beta
    relu->inputs[3] = other;          // addend
    relu->inputs[4] = bn->inputs[3];  // moving_mean
    relu->inputs[5] = bn->inputs[4];  // moving_var
    bn_add_relu_op->attr_parser(&(relu->attrs));
  }

}  // namespace

Graph FuseBNAddRelu(Graph&& g) {
  const auto& shape_vec   = g.GetAttr<mxnet::ShapeVector>("shape");
  const auto& dtype_vec   = g.GetAttr<nnvm::DTypeVector>("dtype");
  const auto& context_vec = g.GetAttr<ContextVector>("context");
  const auto& idx = g.indexed_graph();
  const auto ne_counter = GetNodeEntryCount(g);
  std::unordered_set<nnvm::NodePtr> to_delete;
  DFSVisit(g.outputs, [&ne_counter, &idx, &shape_vec, &dtype_vec,
                       &context_vec, &to_delete](const nnvm::NodePtr& n) {
    if (IsCompatibleReLU(n) && ne_counter.at(n->inputs[0]) == 1) {
      const nnvm::NodePtr& relu = n;
      // TODO(cfujitsang): Should add_n be separated in multiple elemwise_add to allow this fusion ?
      if (n->inputs[0].node->op() == Op::Get("elemwise_add")) {
        const nnvm::NodePtr& add = n->inputs[0].node;
        auto eid = idx.entry_id(add->inputs[0]);
        auto nid = idx.node_id(add.get());
        if (batchnormaddrelu::IsCompatibleBatchNorm(
              add->inputs[0].node, dtype_vec[eid], shape_vec[eid], context_vec[nid]) &&
            add->inputs[0].index == 0 &&
            ne_counter.at(add->inputs[0]) == 1) {
          const nnvm::NodePtr& bn = add->inputs[0].node;
          to_delete.insert(add);
          to_delete.insert(bn);
          FuseBNAddReluNode(bn, add, relu, add->inputs[1]);
        }
        // TODO(cfujitsang): because of topology modification
        // right now we can't merge if the BN is on the inputs[1].
      }
    }
  });
  return g;
}

}  // namespace exec
}  // namespace mxnet
