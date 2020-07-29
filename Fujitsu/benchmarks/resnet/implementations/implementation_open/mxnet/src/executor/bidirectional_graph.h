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
 * \file bidirectional_graph.h
 * \brief Bidirectional graph data structure
 * \author Clement Fuji Tsang, Przemyslaw Tredak
 */
#ifndef MXNET_EXECUTOR_BIDIRECTIONAL_GRAPH_H_
#define MXNET_EXECUTOR_BIDIRECTIONAL_GRAPH_H_

#include <mxnet/base.h>

#include <vector>
#include <algorithm>
#include <deque>
#include <utility>

namespace mxnet {
namespace exec {

namespace {
struct _dummy {};
}

/*!
 * \brief Custom graph class, which contains bi-directional nodes
 * required for traversing in both directions (from outputs to inputs
 * and vice versa). It is a non-owning layer on top of NNVM graph, since
 * NNVM graph enables traversing only in 1 direction (from outputs to inputs).
 */
template <typename DataStruct = _dummy>
class BidirectionalGraph {
 public:
  struct Node;

  struct EdgeRef {
    Node* src_node;
    index_t src_id;
    Node* dst_node;
    index_t dst_id;

    EdgeRef() :
      src_node(nullptr), src_id(-1), dst_node(nullptr), dst_id(-1) {}
    EdgeRef(Node* src, index_t src_id, Node* dst, index_t dst_id) :
      src_node(src), src_id(src_id), dst_node(dst), dst_id(dst_id) {}

    bool operator==(const EdgeRef& other) {
      return (src_node == other.src_node) &&
             (src_id == other.src_id) &&
             (dst_node == other.dst_node) &&
             (dst_id == other.dst_id);
    }
  };

  struct Node {
    nnvm::Node* nnvmptr;
    std::vector<EdgeRef> inputs;
    std::vector<std::vector<EdgeRef>> outputs;
    DataStruct data;

    Node() = default;
    Node(nnvm::Node* nptr, std::vector<EdgeRef> inps,
         std::vector<std::vector<EdgeRef>> outs, DataStruct ds):
      nnvmptr(nptr), inputs(inps), outputs(outs), data(ds) { }

    std::vector<Node*> GetDataConsumers() const {
      std::vector<Node*> ret;

      for (const auto& out : outputs) {
        for (const auto& edge : out) {
          if (edge.dst_node != nullptr) {
            if (std::find(ret.begin(), ret.end(), edge.dst_node) == ret.end()) {
              ret.emplace_back(edge.dst_node);
            }
          }
        }
      }

      return ret;
    }

    std::vector<Node*> GetDataProducers() const {
      std::vector<Node*> ret;

      for (const auto& edge : inputs) {
        if (edge.src_node != nullptr) {
          if (std::find(ret.begin(), ret.end(), edge.src_node) == ret.end()) {
            ret.emplace_back(edge.src_node);
          }
        }
      }

      return ret;
    }
  };

  explicit BidirectionalGraph<DataStruct>(nnvm::Graph* g) {
    auto& idx = g->indexed_graph();
    auto num_nodes = idx.num_nodes();
    nodes_.reserve(num_nodes);
    nnvm2nid.reserve(num_nodes);
    outputs.reserve(idx.outputs().size());
    nnvm_outputs = &(g->outputs);
    // Create all the nodes in a new graph from
    // nodes in the NNVM graph and store them
    // in nodes array
    DFSVisit(g->outputs, [this](const nnvm::NodePtr& n) {
      MakeNode(n);
    });
    // Create all connections between nodes in
    // the graph (both directions)
    for (const auto& it : nnvm2nid) {
      nnvm::Node* nnvmnode = it.first;
      uint32_t nid = it.second;
      for (size_t i = 0; i < nnvmnode->inputs.size(); ++i) {
        const auto& entry = nnvmnode->inputs[i];
        uint32_t input_nid = nnvm2nid[entry.node.get()];

        nodes_[input_nid]->outputs[entry.index].emplace_back(nodes_[input_nid].get(), entry.index,
                                                             nodes_[nid].get(), i);
        nodes_[nid]->inputs[i] = {nodes_[input_nid].get(), static_cast<index_t>(entry.index),
                                  nodes_[nid].get(), static_cast<index_t>(i)};
      }
    }
    // Create output connections from the graph
    for (auto& e : g->outputs) {
      uint32_t nid = nnvm2nid[e.node.get()];
      nodes_[nid]->outputs[e.index].emplace_back(nodes_[nid].get(), e.index,
                                                 nullptr, outputs.size());
      outputs.emplace_back(nodes_[nid].get(), e.index, nullptr, outputs.size());
    }
  }

  // Graph manipulation

  /* \brief Insert a node between two other nodes.
   */
  Node* InsertNode(const nnvm::NodePtr node, const EdgeRef& edge) {
    Node* src = edge.src_node;
    Node* dst = edge.dst_node;

    auto& new_node = MakeNode(node);
    CHECK_EQ(new_node.outputs.size(), 1)
      << "Can insert only nodes with a single output";
    CHECK_EQ(new_node.inputs.size(), 1)
      << "Can insert only nodes with a single input";

    const EdgeRef ref_edge = edge;

    // Update bidirectional graph
    for (auto& e : src->outputs[ref_edge.src_id]) {
      if (e == ref_edge) {
        e.dst_node = &new_node;
        e.dst_id = 0;
        new_node.inputs[0] = e;
        break;
      }
    }


    if (dst != nullptr) {
      for (auto& e : dst->inputs) {
        if (e == ref_edge) {
          e.src_node = &new_node;
          e.src_id = 0;
          new_node.outputs[0].emplace_back(e);
          break;
        }
      }
    } else {
      EdgeRef e = ref_edge;
      e.src_node = &new_node;
      e.src_id = 0;
      new_node.outputs[0].emplace_back(e);
      outputs[ref_edge.dst_id] = e;
    }

    // Update underlying NNVM graph
    nnvm::NodeEntry& entry_to_update = (dst != nullptr) ? dst->nnvmptr->inputs[ref_edge.dst_id] :
                                                          (*nnvm_outputs)[ref_edge.dst_id];
    nnvm::NodeEntry dst_entry = entry_to_update;
    nnvm::NodeEntry new_entry(node, 0, dst_entry.version);
    entry_to_update = std::move(new_entry);
    node->inputs.resize(1);
    node->inputs[0] = std::move(dst_entry);

    return &new_node;
  }

  /* \brief Delete a node.
   */
  void DeleteNode(const Node& node) {
    CHECK(node.inputs.size() == 1 && node.outputs.size() == 1) <<
      "Only a node with a single input and output may be deleted";
    const EdgeRef& input_edge = node.inputs[0];
    std::vector<EdgeRef> output_edges = node.outputs[0];

    Node* src_node = input_edge.src_node;
    auto& src_node_outputs = src_node->outputs[input_edge.src_id];

    for (auto& edge : output_edges) {
      edge.src_node = src_node;
      edge.src_id = edge.src_id;
    }

    std::remove(src_node_outputs.begin(),
                src_node_outputs.end(),
                input_edge);
    src_node_outputs.insert(src_node_outputs.end(),
                            output_edges.begin(),
                            output_edges.end());

    const nnvm::NodeEntry& nnvm_entry = node.nnvmptr->inputs[0];

    for (const auto& edge : output_edges) {
      if (edge.dst_node != nullptr) {
        edge.dst_node->inputs[edge.dst_id] = edge;
        edge.dst_node->nnvmptr->inputs[edge.dst_id] = nnvm_entry;
      } else {
        outputs[edge.dst_id] = edge;
        (*nnvm_outputs)[edge.dst_id] = nnvm_entry;
      }
    }
  }

  std::pair<Node*, nnvm::NodePtr> CopyNode(Node* node) {
    nnvm::NodePtr nnvm_node = nnvm::Node::Create();
    nnvm_node->attrs = node->nnvmptr->attrs;
    return std::make_pair(&(MakeNode(nnvm_node)), nnvm_node);
  }

  Node* DetachEdge(const EdgeRef& e) {
    Node* new_node;
    nnvm::NodePtr new_nnvm_node;
    std::tie(new_node, new_nnvm_node) = CopyNode(e.src_node);

    // Copy input edges
    CHECK_EQ(new_node->inputs.size(), e.src_node->inputs.size());
    for (size_t i = 0; i < new_node->inputs.size(); ++i) {
      const auto& edge = e.src_node->inputs[i];
      new_node->inputs[i] = {edge.src_node, edge.src_id,
                             new_node, edge.dst_id};
    }
    new_nnvm_node->inputs = e.src_node->nnvmptr->inputs;

    // Rewire output edges
    std::remove(e.src_node->outputs[e.src_id].begin(),
                e.src_node->outputs[e.src_id].end(),
                e);
    if (e.dst_node != nullptr) {
      e.dst_node->inputs[e.dst_id].src_node = new_node;
      e.dst_node->nnvmptr->inputs[e.dst_id].node = new_nnvm_node;
    } else {
      outputs[e.dst_id].src_node = new_node;
      (*nnvm_outputs)[e.dst_id].node = new_nnvm_node;
    }
    CHECK_EQ(new_node->outputs.size(), e.src_node->outputs.size());
    new_node->outputs[e.src_id].emplace_back(new_node, e.src_id,
                                             e.dst_node, e.dst_id);
    return new_node;
  }

  /* \brief Get all subsets of nodes, where:
   *  - graph constructed from nodes in each subset is a connected graph
   *  - every node fulfills a predicate is_compatible
   *  - if nodes u and v are part of a subset, then for each path between
   *    u and v in the original directed graph, all nodes on those paths
   *    are also part of the subset
   * \param is_compatible A function taking nnvm::Node* and returning bool
   *                      which identifies which nodes should be included in
   *                      subsets.
   */
  template<typename FCompatible>
  std::vector<std::unordered_set<Node*>> get_subsets(FCompatible is_compatible) {
    std::vector<std::unordered_set<Node*>> subgraphs;
    std::unordered_set<Node*> incomp_set;
    std::vector<std::pair<bool, PairSet>> separation_sets;
    // Check each node for compatibility
    // and, if it is incompatible, mark nodes
    // on each side of it as not possible to be
    // in the same subset
    for (const auto& node : nodes_) {
      if (!is_compatible(node->nnvmptr)) {
        incomp_set.insert(node.get());
      }
    }
    for (auto& node_ptr : nodes_) {
      Node& node = *node_ptr;
      if (incomp_set.count(&node) != 0) {
        // Check if all your inputs are incompatible too.
        // If so, then your separation set does not matter,
        // because it will covered by the sets of your inputs
        bool inside_node = true;
        for (const auto& input_edge : node.inputs) {
          if (incomp_set.count(input_edge.src_node) == 0) {
            inside_node = false;
          }
        }
        if (!inside_node) {
          std::unordered_set<Node*> in_graph;
          std::unordered_set<Node*> out_graph;
          std::vector<Node*> dummy_head;
          dummy_head.emplace_back(&node);
          DFS(dummy_head, false, [&out_graph](Node* node) {
              out_graph.insert(node);
          });
          DFS(dummy_head, true, [&in_graph](Node* node) {
              in_graph.insert(node);
          });
            separation_sets.push_back(std::make_pair(true,
                                                     std::make_pair(in_graph, out_graph)));
        } else {
          separation_sets.push_back(std::make_pair(false, PairSet()));
        }
      } else {
        separation_sets.push_back(std::make_pair(false, PairSet()));
      }
    }
    IncompMap incomp_map;
    // For each node construct the map of nodes that cannot be in
    // the same subset
    index_t num_nodes = nodes_.size();
    for (index_t i = 0; i < num_nodes; ++i) {
      const auto n = nodes_[i].get();
      if (incomp_set.count(n) == 0) {
        for (index_t j = i + 1; j < num_nodes; ++j) {
          const auto& sep_set_pair = separation_sets[j];
          if (sep_set_pair.first && incomp_map[n].count(nodes_[j].get()) == 0) {
            const auto& p = sep_set_pair.second;
            if (p.first.count(n)) {
              incomp_map[n].insert(p.second.begin(), p.second.end());
            } else if (p.second.count(n)) {
              incomp_map[n].insert(p.first.begin(), p.first.end());
            }
          }
        }
        for (index_t j = i - 1; j >= 0; --j) {
          const auto& sep_set_pair = separation_sets[j];
          if (sep_set_pair.first && incomp_map[n].count(nodes_[j].get()) == 0) {
            const auto& p = sep_set_pair.second;
            if (p.first.count(n)) {
              incomp_map[n].insert(p.second.begin(), p.second.end());
            } else if (p.second.count(n)) {
              incomp_map[n].insert(p.first.begin(), p.first.end());
            }
          }
        }
        for (Node* incomp_n : incomp_set) {
          incomp_map[n].erase(incomp_n);
        }
      }
    }
    std::unordered_set<Node*> unused_set;

    for (auto& n : nodes_) {
      if (incomp_set.count(n.get()) == 0) {
        unused_set.insert(n.get());
      }
    }
    std::unordered_set<Node*> visited;
    std::deque<Node*> queue;
    for (const auto& out : outputs) {
      queue.emplace_back(out.src_node);
    }
    // Create subsets
    while (!queue.empty()) {
      Node* vertex = queue.front();
      queue.pop_front();
      if (!visited.count(vertex)) {
        visited.insert(vertex);
        if (unused_set.count(vertex)) {
          subgraphs.emplace_back(naive_grow_subgraph(vertex, &unused_set, &incomp_map));
        }
        for (const EdgeRef& input : vertex->inputs) {
          if (input.src_node != nullptr) {
            queue.emplace_back(input.src_node);
          }
        }
      }
    }
    return subgraphs;
  }

  std::vector<std::unique_ptr<Node>>& nodes() {
    return nodes_;
  }

 private:
  using PairSet = std::pair<std::unordered_set<Node*>, std::unordered_set<Node*>>;
  using PairVec = std::pair<std::vector<Node*>, std::vector<Node*>>;
  using IncompMap = std::unordered_map<Node*, std::unordered_set<Node*>>;

  Node& MakeNode(const nnvm::NodePtr& n) {
    nnvm2nid[n.get()] = static_cast<uint32_t>(nodes_.size());
    nodes_.emplace_back();
    nodes_.back().reset(new Node());
    Node& new_node = *(nodes_.back());
    new_node.nnvmptr = n.get();
    if (n->num_inputs() == nnvm::kVarg || n->is_variable()) {
      new_node.inputs.resize(n->inputs.size());
    } else {
      new_node.inputs.resize(n->num_inputs());
    }
    if (!n->is_variable() && n->inputs.size() != 0) {
      CHECK_EQ(new_node.inputs.size(), n->inputs.size())
        << "Number of inputs to operator " << n->op()->name << " (" << n->num_inputs()
        << ") does not match the actual number of inputs provided to operator "
        << n->attrs.name << " (" << n->inputs.size() << ").";
    }
    new_node.outputs.resize(n->num_outputs());
    return new_node;
  }

  /* \brief Traverse the graph using DFS in either direction.
   * \param heads Starting nodes for the DFS algorithm.
   * \param reverse If true, DFS will traverse the graph from
   *                outputs to inputs. Otherwise, it will
   *                traverse the graph from inputs to outputs.
   * \param fvisit Function to call on each visisted node.
   */
  template <typename FVisit>
  void DFS(const std::vector<Node*>& heads, bool reverse, FVisit fvisit) {
    std::unordered_set<Node*> visited;
    std::vector<Node*> vec(heads.begin(), heads.end());
    visited.reserve(heads.size());
    while (!vec.empty()) {
      Node* vertex = vec.back();
      vec.pop_back();
      if (visited.count(vertex) == 0) {
        visited.insert(vertex);
        fvisit(vertex);
        std::vector<Node*> nexts = reverse ? vertex->GetDataProducers()
                                           : vertex->GetDataConsumers();
        for (Node* node : nexts) {
          if (visited.count(node) == 0) {
            vec.emplace_back(node);
          }
        }
      }
    }
  }

  /* \brief Get the connected subgraph that contains the head node,
   *        only previously unused nodes, according to the rules
   *        from incompatibility map.
   * \param head Node which needs to be part of the returned subgraph.
   * \param unused_set Only nodes from this set will be considered when
   *                   adding to the growing subgraph.
   * \param incomp_map Map containing data on which nodes are incompatible
   *                   to be in the same subgraph.
   */
  std::unordered_set<Node*> naive_grow_subgraph(Node* head,
                                                std::unordered_set<Node*>* unused_set,
                                                IncompMap* incomp_map) {
    std::unordered_set<Node*> subgraph;
    std::unordered_set<Node*> incomp_set;
    std::vector<Node*> stack;
    stack.emplace_back(head);
    while (!stack.empty()) {
      Node* vertex = stack.back();
      stack.pop_back();
      if (unused_set->count(vertex) && !incomp_set.count(vertex)) {
        unused_set->erase(vertex);
        subgraph.insert(vertex);
        incomp_set.insert((*incomp_map)[vertex].begin(), (*incomp_map)[vertex].end());
        for (Node* input : vertex->GetDataProducers()) {
          if (unused_set->count(input) && !incomp_set.count(input)) {
            stack.emplace_back(input);
          }
        }
        for (Node* output : vertex->GetDataConsumers()) {
          if (unused_set->count(output) && !incomp_set.count(output)) {
            stack.emplace_back(output);
          }
        }
      }
    }
    return subgraph;
  }

  friend class nnvm::Graph;

  std::vector<std::unique_ptr<Node>> nodes_;
  std::unordered_map<nnvm::Node*, uint32_t> nnvm2nid;
  std::vector<EdgeRef> outputs;
  std::vector<nnvm::NodeEntry>* nnvm_outputs;
};  // class BidirectionalGraph

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_BIDIRECTIONAL_GRAPH_H_
