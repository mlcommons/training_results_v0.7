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
 * \file simple_partition_pass.cc
 * \brief
 * \author Clement Fuji Tsang
 */

#include "./simple_partition_pass.h"


namespace mxnet {
namespace exec {


nnvm::NodeEntryMap<uint32_t> GetSubgraphOutputs(Graph g, NodeRawPtrSet subgraph_set) {
  nnvm::NodeEntryMap<uint32_t> outputs;
  uint32_t count = 0;
  for (auto& e : g.outputs) {
    if (subgraph_set.count(e.node.get()) && !outputs.count(e)) {
      outputs.insert({e, count++});
    }
  }
  DFSVisit(g.outputs, [&subgraph_set, &outputs, &count](const nnvm::NodePtr &node){
    if (!subgraph_set.count(node.get())) {
      for (auto& e : node->inputs) {
        if (subgraph_set.count(e.node.get()) && !outputs.count(e)) {
          outputs.insert({e, count++});
        }
      }
    }
  });
  return outputs;
}


std::vector<nnvm::NodeEntry> GetSubgraphInputs(Graph g, NodeRawPtrSet subgraph_set) {
  std::vector<nnvm::NodeEntry> inputs;
  nnvm::NodeEntryMap<nnvm::NodeEntry> entry_map;
  DFSVisit(g.outputs, [&subgraph_set, &inputs, &entry_map](const nnvm::NodePtr &node){
    if (subgraph_set.count(node.get())) {
      for (auto &e : node->inputs) {
        if (!subgraph_set.count(e.node.get())) {
          if (entry_map.count(e)) {
            e = entry_map[e];
          } else {
            auto new_node = nnvm::Node::Create();
            new_node->attrs.name = "input_" + std::to_string(inputs.size());
            entry_map.insert({e, nnvm::NodeEntry{new_node, 0, 0}});
            inputs.push_back(e);
            e.node = new_node;
            e.index = 0;
          }
        }
      }
    }
  });
  // Fix ordering of w.r.t to topology
  Graph _g;
  _g.outputs = g.outputs;
  const auto &idx = _g.indexed_graph();
  std::sort(inputs.begin(), inputs.end(),
      [&idx, &entry_map](const nnvm::NodeEntry lhs, const nnvm::NodeEntry rhs) {
        return idx.entry_id(entry_map.at(lhs)) < idx.entry_id(entry_map.at(rhs));
      });
  return inputs;
}


std::unordered_map<uint32_t, uint32_t> GetGraphInputsMap(const Graph& g) {
  std::unordered_map<uint32_t, uint32_t> outputs;
  auto& idx = g.indexed_graph();
  outputs.reserve(idx.num_nodes());
  std::vector<uint32_t> input_nodes = idx.input_nodes();
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    outputs[input_nodes[i]] = static_cast<uint32_t>(i);
  }
  return outputs;
}


}  // namespace exec
}  // namespace mxnet
