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
 * \file alm.h
 * \brief Automatic Layout Manager
 * \author Dawid Tracz
 */

#ifndef MXNET_EXECUTOR_ALM_H_
#define MXNET_EXECUTOR_ALM_H_

#include <nnvm/graph.h>
#include <nnvm/node.h>
#include <nnvm/symbolic.h>
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <set>
#include <queue>
#include "./exec_pass.h"
#include "./simple_partition_pass.h"



namespace mxnet {
namespace exec {


class AutomaticLayoutManager {
 public:
  enum passdir_t {kNONE, kFWD, kBWD};

  struct NodeInfo {
    bool blacklisted = false;
    passdir_t pass_direction;
    nnvm::TShape axes;
    bool is_added = false;

    NodeInfo(): pass_direction(kNONE) { }
    NodeInfo(passdir_t pd, nnvm::TShape ax, bool is_added): pass_direction(pd), axes(ax),
                                                            is_added(is_added) { }
  };

  using BG = BidirectionalGraph<NodeInfo>;
  using BGNode = BG::Node;
  using BGEdge = BG::EdgeRef;


  const std::string name_base_ = "ALM_transpose_";
  nnvm::Graph graph_;
  BidirectionalGraph<NodeInfo> bigraph_;


  explicit AutomaticLayoutManager(const nnvm::Symbol& symbol):
    graph_(symbol2Graph(symbol)),
    bigraph_(&graph_) { }

  AutomaticLayoutManager(AutomaticLayoutManager&) = delete;
  AutomaticLayoutManager(AutomaticLayoutManager&&) = delete;
  AutomaticLayoutManager operator=(AutomaticLayoutManager&) = delete;
  AutomaticLayoutManager operator=(AutomaticLayoutManager&&) = delete;


  void run(std::unordered_map<std::string, std::string> targets);

  nnvm::Symbol GetOptimizedSymbol() const {
    nnvm::Symbol s;
    s.outputs = graph_.outputs;
    return s.Copy();
  }

 private:
  std::unordered_map<std::string, mshadow::LayoutFlag> targets_;
  std::unordered_set<std::shared_ptr<BGNode>> newNodes_;

  unsigned n_newNodes = 0;


  std::string getNextName() {
    return name_base_ + std::to_string(n_newNodes++);
  }

  nnvm::Graph symbol2Graph(const nnvm::Symbol &s) {
    nnvm::Graph g;
    g.outputs = s.outputs;
    return g;
  }


  nnvm::NodePtr CreateTransposeNode(const NodeInfo& info);

  /*!
   * \brief Repins all inputs and outputs excluding it from the graph,
   *        both for BGNode as for eventually nnvm::Node inside.
   *        Then marks node as blacklisted.
   * \param node Node that's to be deleted.
   */
  void deleteNode(BGNode* node);


  /*!
   * \brief Changes layout of given node (this node has to have "layout" attribute)
   *        and surrounds it with proper transposes.
   * \param node Pointer to node that layout has to be changed.
   * \return Set of pointers to created Transposes or empty set if function failed,
   *         and no tranpose was created.
   */
  std::set<BGNode*> changeLayout(BGNode* node, mshadow::LayoutFlag layout);


  /*!
   * \brief If possible passes transpose further (forward or backward)
   *        by removing it and creating another on all the others edges
   *        of the next/previous node, or merge with it if it's also transpose.
   * \param t_ptr Pointer to the transpose node that's need to be passed.
   * \return Set of pointers to created Transposes, empty set if the tranpose
   *         collapsed with another, empty set if transpose collapsed to identity
   *         with the next or {t_ptr} if function failed and t_ptr was not removed.
   */
  std::set<BGNode*> passTranspose(BGNode* t_ptr);


  /*!
   * \brief Merge two adjacent transpose nodes.
   * \param first Pointer to the first transpose node.
   * \param second Pointer to the second transpose node.
   * \return Pointer to eventually created node, or nullptr
   *         if none was created.
   */
  BGNode* mergeTransposes(BGNode* first, BGNode* second);


  /*!
   * \brief Create new nnvm::Node with transpose operation,
   *        put it in given node and register it.
   * \param node Empty BidirectionalGraph<NodeInfo>::Node.
   */
  void addTransposeOp(BGNode* node);


  /*!
   * \brief Inserts new ALM transpose node in between two given.
   * \param edge Edge on which transpose will be inserted.
   * \param newInfo NodeInfo for the new transpose node.
   * \return Pointer tho the node that was created.
   */
  BGNode* insertTransposeBetween(const BGEdge* edge, const NodeInfo& newInfo);


  /*!
   * \brief Surrounds given node with transposes, according to given axes.
   * \param node Node to be surrounded.
   * \param inpAxes Vector of axes for input transposes.
   * \param outAxes Vector of axes for output transposes.
   * \return Set of pointers to created transposes.
   */
  std::set<BGNode*> surroundNode(BGNode* node,
                                 std::vector<nnvm::TShape>* inpAxes,
                                 std::vector<nnvm::TShape>* outAxes);
};

}  // namespace exec
}  // namespace mxnet

#endif  // MXNET_EXECUTOR_ALM_H_
