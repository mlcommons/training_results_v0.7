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
 * \file simple_partition_pass.h
 * \brief Simple pass for partitioning a graph.
 * \author Clement Fuji Tsang
 */
#ifndef MXNET_EXECUTOR_SIMPLE_PARTITION_PASS_H_
#define MXNET_EXECUTOR_SIMPLE_PARTITION_PASS_H_

#include <mxnet/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <nnvm/graph_attr_types.h>
#include <utility>
#include <deque>
#include <algorithm>
#include <vector>

#include "exec_pass.h"
#include "bidirectional_graph.h"

namespace mxnet {
namespace exec {

using NodeEntrySet = std::unordered_set<nnvm::NodeEntry, nnvm::NodeEntryHash,
                                        nnvm::NodeEntryEqual>;
using NodeRawPtrSet = std::unordered_set<nnvm::Node*>;

/*!
 * \brief Get the output nodes of the subgraph in the main graph.
 * \return a map between the node in the main graph and the output index of the subgraph node
*/
nnvm::NodeEntryMap<uint32_t> GetSubgraphOutputs(Graph g, NodeRawPtrSet subgraph_set);


/*!
 * \brief Create new input nodes of the subgraph and plug them.
 * \return the inputs of the subgraph node in the main graph
*/
std::vector<nnvm::NodeEntry> GetSubgraphInputs(Graph g, NodeRawPtrSet subgraph_set);


std::unordered_map<uint32_t, uint32_t> GetGraphInputsMap(const Graph& g);


/*!
 * \brief Helper function to display what nodes are in a specific subset.
 */
inline void dispNodesSet(Graph g, NodeRawPtrSet s) {
  DFSVisit(g.outputs, [&s](const nnvm::NodePtr n) {
    if (s.count(n.get())) {
      std::cout << "  Y " << n->attrs.name << std::endl;
    } else {
      std::cout << "  N " << n->attrs.name << std::endl;
    }
  });
}

/*!
 * \brief Replace a set of nodes by a subgraph node.
 */
template<typename FCreateNode>
Graph ReplaceSubgraphs(Graph&& g, const std::vector<NodeRawPtrSet>& subgraph_sets,
                       FCreateNode create_subgraph_node) {
  for (auto subgraph_set : subgraph_sets) {
    // Create MXNet subgraph
    Graph subgraph;
    const auto sub_outputs_in_main = GetSubgraphOutputs(g, subgraph_set);
    subgraph.outputs.resize(sub_outputs_in_main.size());
    for (auto p : sub_outputs_in_main) {
      subgraph.outputs[p.second] = p.first;
    }
    // To generate a subgraph an input has to be replaced by data node (no op)
    // and it has to be agnostic to the node from which it's an output
    // (For example, even if two inputs are two different outputs from the same node,
    // they need to be replaced by two completely separate data nodes)
    auto inputs = GetSubgraphInputs(subgraph, subgraph_set);
    auto subgraph_node = create_subgraph_node(subgraph);
    subgraph_node->inputs = inputs;
    // replug inputs of node out of subgraph to be output of the subgraph node
    // if it was a node in the subgraph
    DFSVisit(g.outputs,
        [&subgraph_node, &subgraph_set, &sub_outputs_in_main](const nnvm::NodePtr node) {
      if (!subgraph_set.count(node.get())) {
        for (auto &e : node->inputs) {
          auto it = sub_outputs_in_main.find(e);
          if (it != sub_outputs_in_main.end()) {
            e.node = subgraph_node;
            e.index = it->second;
          }
        }
      }
    });
    // replug outputs of the graph to be output of the subgraph node
    // if it was a node in the subgraph
    for (auto &e : g.outputs) {
      auto it = sub_outputs_in_main.find(e);
      if (it != sub_outputs_in_main.end()) {
        e.node = subgraph_node;
        e.index = it->second;
      }
    }
    // move control dependencies between nodes of the subgraph and out of the subgraph
    // to a dependencies between the subgraph node and the nodes out of the subgraph
    DFSVisit(g.outputs, [&subgraph_node, &subgraph_set](const nnvm::NodePtr& node) {
      for (auto &e : node->control_deps) {
        if (subgraph_set.count(e.get()))
          e = subgraph_node;
      }
    });
    DFSVisit(subgraph.outputs, [&subgraph_node, &subgraph_set](const nnvm::NodePtr& node) {
      auto it = node->control_deps.begin();
      while (it != node->control_deps.end()) {
        if (subgraph_set.count(it->get())) {
          ++it;
        } else {
          subgraph_node->control_deps.push_back(*it);
          it = node->control_deps.erase(it);
        }
      }
    });
  }
  Graph new_graph;
  new_graph.outputs = g.outputs;
  return new_graph;
}

/* \brief Get all subsets of nodes, where:
 *  - graph constructed from nodes in each subset is a connected graph
 *  - every node fulfills a predicate is_compatible
 *  - if nodes u and v are part of a subset, then for each path between
 *    u and v in the original directed graph, all nodes on those paths
 *    are also part of the subset
 * \param g NNVM graph
 * \param is_compatible A function taking nnvm::Node* and returning bool
 *                      which identifies which nodes should be included in
 *                      subsets.
 */
template<typename FCompatible>
std::vector<NodeRawPtrSet> GetCompatibleSubsets(Graph* g, FCompatible is_compatible) {
  BidirectionalGraph<> biG(g);
  std::vector<std::unordered_set<BidirectionalGraph<>::Node*>> subsets =
    biG.get_subsets(is_compatible);
  std::vector<NodeRawPtrSet> nnvm_subsets;
  nnvm_subsets.reserve(subsets.size());
  for (auto& subset : subsets) {
    if (subset.size() > 1) {
      NodeRawPtrSet node_set;
      node_set.reserve(subset.size());
      for (auto& n : subset) {
        node_set.insert(n->nnvmptr);
      }
      nnvm_subsets.push_back(node_set);
    }
  }
  return nnvm_subsets;
}

}  // namespace exec
}  // namespace mxnet
#endif  // MXNET_EXECUTOR_SIMPLE_PARTITION_PASS_H_
