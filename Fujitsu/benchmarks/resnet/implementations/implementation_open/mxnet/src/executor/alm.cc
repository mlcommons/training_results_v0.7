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
 * \file alm.cc
 * \brief Automatic Layout Manager
 * \author Dawid Tracz
 */

#include "./alm.h"
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include "../operator/operator_common.h"


namespace mxnet {
namespace exec {

using ALM = AutomaticLayoutManager;
using BGNode = ALM::BGNode;

namespace {

static inline bool isNative(const BGNode& node) {
  return node.data.is_added == false;
}

static inline void addAllTo(std::queue<BGNode*>* queue, std::set<BGNode*>&& set) {
  for (auto ndPtr : set) {
    queue->push(ndPtr);
  }
}

template <typename Iterable>
static inline bool isIdentity(Iterable arr) {
  int i = 0;
  for (const auto& elem : arr) {
    if (elem != i++)
      return false;
  }
  return true;
}

/*!
 * \brief Determines pass direction of transpose node created by collapsing two other.
 * \param t1 Pass direction of first transpose node.
 * \param t2 Pass direction of second transpose node.
 * \return Pass direction for new node.
 */
static inline ALM::passdir_t mergeDirs(ALM::passdir_t t1, ALM::passdir_t t2) {
  auto opposite = [](ALM::passdir_t t) {
    switch (t) {
     case ALM::kBWD:
      return ALM::kFWD;
     case ALM::kFWD:
      return ALM::kBWD;
     default:
      return ALM::kNONE;
    }
  };
  if (t1 == opposite(t2))
    return ALM::kNONE;
  else
    return t1 != ALM::kNONE ? t1 : t2;
}

}  // namespace


/*!
 * \brief Applies one transpose axes on the top of another.
 *        Results in axes of a transpose created by merging the transposes
 *        represented by given axes.
 * \param ax1 First transpose axes.
 * \param ax2 Second transpose axes. Those are to be applied on the first.
 * \return Axes of result transpose.
 */
static inline nnvm::TShape apply(const nnvm::TShape& ax1, const nnvm::TShape& ax2) {
  nnvm::TShape axes = ax1;
  CHECK_EQ(ax1.ndim(), ax2.ndim()) << "Axes ndim missmatch";
  for (size_t i = 0; i < ax1.ndim(); i++) {
    axes[i] = ax1[ax2[i]];
  }
  return axes;
}

nnvm::NodePtr ALM::CreateTransposeNode(const NodeInfo& info) {
  nnvm::NodePtr newptr = nnvm::Node::Create();
  newptr->attrs.op = nnvm::Op::Get("transpose");
  newptr->attrs.name = getNextName();
  // set tranpose axes
  std::stringstream ss;
  ss << info.axes;
  newptr->attrs.dict["axes"] = ss.str();
  newptr->op()->attr_parser(&(newptr->attrs));
  return newptr;
}


void ALM::run(std::unordered_map<std::string, std::string> targets) {
  targets_.clear();
  for (auto pair : targets) {
    auto targetLayout = mshadow::layoutFlag(pair.second);
    if (targetLayout != mshadow::kUNKNOWN)
      this->targets_[pair.first] = targetLayout;
    else
      LOG(WARNING) << "Unknown layout: " << pair.second << ". ignoring...";
  }
  // surround targets with transposes
  std::queue<BGNode*> newTransps;
  size_t num_nodes = bigraph_.nodes().size();
  for (size_t i = 0; i < num_nodes; ++i) {
    BGNode& node = *(bigraph_.nodes()[i]);
    if (node.nnvmptr->op() != nullptr &&
        targets_.find(node.nnvmptr->op()->name) != targets_.end()) {
      auto targetLayout = targets_[node.nnvmptr->op()->name];
      addAllTo(&newTransps, changeLayout(&node, targetLayout));
    }
  }
  // push transposes through a graph
  std::vector<BGNode*> ready;
  while (!newTransps.empty()) {
    BGNode* t = newTransps.front();
    newTransps.pop();
    if (t->data.blacklisted) {
      continue;
    }
    if (t->data.pass_direction != kNONE) {
      addAllTo(&newTransps, passTranspose(t));
    } else {
      ready.push_back(t);
    }
  }
  // eliminates identity transposes
  for (auto t : ready) {
    if (isIdentity(t->data.axes)) {
      deleteNode(t);
    }
  }
}

void ALM::deleteNode(BGNode* node) {
  bigraph_.DeleteNode(*node);
  node->data.blacklisted = true;
}


BGNode* ALM::mergeTransposes(BGNode* first, BGNode* second) {
  auto getAxes = [](BGNode* node) {
    if (node->nnvmptr) {
      CHECK_EQ(node->nnvmptr->op()->name, "transpose");
      auto& dict = node->nnvmptr->attrs.dict;
      CHECK(dict.find("axes") != dict.end()) << "Transpose node should have \"axes\" param";
      std::stringstream ss;
      ss << dict["axes"];
      ss >> node->data.axes;
    }
    return node->data.axes;
  };
  CHECK_EQ(first->inputs.size(), 1) << "Transpose node should have exactly one input";
  CHECK_EQ(second->inputs.size(), 1) << "Transpose node should have exactly one input";
  // let first be the father of second
  if (first != second->inputs[0].src_node)
    std::swap(first, second);
  CHECK_EQ(first, second->inputs[0].src_node) << "Only adjacent nodes can be merged";
  const auto& ax1 = getAxes(first);
  const auto& ax2 = getAxes(second);
  CHECK_EQ(ax1.ndim(), ax2.ndim()) << "Axes size mismatch";
  nnvm::TShape axes = apply(ax1, ax2);
  // get future pass_direction
  auto newPassDir = mergeDirs(first->data.pass_direction,
                              second->data.pass_direction);

  BGNode* node_to_delete;
  if (first->outputs[0].size() > 1) {
    node_to_delete = bigraph_.DetachEdge(second->inputs[0]);
  } else {
    node_to_delete = first;
  }
  deleteNode(node_to_delete);

  // now modify second's axes and if result is identity -- delete it
  if (isIdentity(axes)) {
    deleteNode(second);
    return nullptr;
  } else {
    std::stringstream ss;
    ss << axes;
    second->nnvmptr->attrs.dict["axes"] = ss.str();
    second->nnvmptr->op()->attr_parser(&(second->nnvmptr->attrs));
    second->data = NodeInfo(newPassDir, axes, true);
    return second;
  }
}


std::set<BGNode*> ALM::passTranspose(BGNode* t_ptr) {
  CHECK_EQ(t_ptr->inputs.size(), 1) << "Transpose node should have exactly one input";
  CHECK_EQ(t_ptr->outputs.size(), 1) << "Transpose node should have exactly one output";
  CHECK_EQ(t_ptr->outputs[0].size(), 1) << "Transpose node should have exactly one consumer";
  // get node to pass transpose through
  BGNode* node;
  index_t index;
  if (t_ptr->data.pass_direction == kFWD) {
    node = t_ptr->outputs[0][0].dst_node;
    index = t_ptr->outputs[0][0].dst_id;
  } else if (t_ptr->data.pass_direction == kBWD) {
    node = t_ptr->inputs[0].src_node;
    index = t_ptr->inputs[0].src_id;
  } else {
    return {t_ptr};
  }
  // if it's an output, skip
  if (node == nullptr) {
    t_ptr->data.pass_direction = kNONE;
    return {t_ptr};
  }
  // if it's a data node, skip
  if (node->nnvmptr && node->nnvmptr->op() == nullptr) {
    t_ptr->data.pass_direction = kNONE;
    return {t_ptr};
  }
  // if it's transpose too -- just merge it
  if ((node->nnvmptr && node->nnvmptr->op()->name == "transpose")) {
    auto newNode = mergeTransposes(t_ptr, node);
    if (newNode != nullptr)
      return {newNode};
    else
      return { };
  }
  // get changeLayout function
  auto opMap = Op::GetAttr<mxnet::alm::FChangeLayout>("FChangeLayout");
  auto changeLayout = opMap.get(node->nnvmptr->op(), nullptr);
  if (changeLayout == nullptr) {
    t_ptr->data.pass_direction = kNONE;
    return {t_ptr};
  }
  // set vectors
  std::vector<nnvm::TShape> inpTransposes(node->inputs.size());
  std::vector<nnvm::TShape> outTransposes(node->outputs.size());
  auto axes = t_ptr->data.axes;
  if (t_ptr->data.pass_direction == kFWD) {
    inpTransposes[index] = common::ReverseTransposeAxes(axes);
  } else if (t_ptr->data.pass_direction == kBWD) {
    outTransposes[index] = common::ReverseTransposeAxes(axes);
  }
  // changeLayout
  changeLayout(&node->nnvmptr->attrs, mshadow::kUNKNOWN,
               &inpTransposes, &outTransposes);
  // if changeLayout fails axes vectors should be cleared
  if (inpTransposes.empty() || outTransposes.empty()) {
    t_ptr->data.pass_direction = kNONE;
    return {t_ptr};
  }
  auto newNodes = surroundNode(node, &inpTransposes, &outTransposes);
  // collapse neighbour transposes
  auto ndIter = newNodes.end();
  if (t_ptr->data.pass_direction == kFWD)
    ndIter = newNodes.find(t_ptr->outputs[0][0].dst_node);
  else
    ndIter = newNodes.find(t_ptr->inputs[0].src_node);
  CHECK(ndIter != newNodes.end()) << "One opposite node should be created";
  auto mergedNd = mergeTransposes(t_ptr, *ndIter);
  CHECK(mergedNd == nullptr) << "Opposite transposes should collapse to nothing";
  newNodes.erase(ndIter);
  if (node->nnvmptr->op()->attr_parser)
    node->nnvmptr->op()->attr_parser(&(node->nnvmptr->attrs));
  return newNodes;
}


BGNode* ALM::insertTransposeBetween(const BGEdge* edge, const NodeInfo& newInfo) {
  nnvm::NodePtr nnvm_node = CreateTransposeNode(newInfo);
  BGNode* new_node = bigraph_.InsertNode(nnvm_node, *edge);
  new_node->data = newInfo;

  return new_node;
}


std::set<BGNode*> ALM::changeLayout(BGNode* node,
                                     mshadow::LayoutFlag targetLayout) {
  if (node->nnvmptr->attrs.dict["layout"] == mshadow::toString(targetLayout))
    return { };
  std::vector<nnvm::TShape> inpTransposes;
  std::vector<nnvm::TShape> outTransposes;
  auto opMap = Op::GetAttr<mxnet::alm::FChangeLayout>("FChangeLayout");
  auto changeLayout = opMap.get(node->nnvmptr->op(), nullptr);
  if (changeLayout == nullptr)
    return { };
  auto srcLayout = changeLayout(&node->nnvmptr->attrs, targetLayout,
                                &inpTransposes, &outTransposes);
  if (srcLayout == mshadow::kUNKNOWN) {
    LOG(WARNING) << "ALM failed to change layout of " << node->nnvmptr->op()->name
                 << " from " << node->nnvmptr->attrs.dict["layout"]
                 << " to " << mshadow::toString(targetLayout);
    return { };
  }
  auto newNodes = surroundNode(node, &inpTransposes, &outTransposes);
  node->nnvmptr->op()->attr_parser(&(node->nnvmptr->attrs));
  return newNodes;
}

std::set<BGNode*> ALM::surroundNode(BGNode* node,
                                     std::vector<nnvm::TShape>* inpAxes,
                                     std::vector<nnvm::TShape>* outAxes) {
  std::set<BGNode*> created;
  if (node->inputs.empty()) {
    LOG(FATAL) << "Node does not have any input";
  } else {
    CHECK_EQ(node->inputs.size(), inpAxes->size());
    for (size_t i = 0; i < inpAxes->size(); i++) {
      if (inpAxes->at(i).ndim() <= 0 || isIdentity(inpAxes->at(i)))
        continue;
      NodeInfo newInfo(kBWD, inpAxes->at(i), true);
      auto ndPtr = insertTransposeBetween(&(node->inputs[i]), newInfo);
      created.insert(ndPtr);
    }
  }
  CHECK(!node->outputs.empty()) << "Node has to have outputs to be able to be surrounded.";
  CHECK(outAxes->size() == node->outputs.size());
  for (size_t i = 0; i < node->outputs.size(); ++i) {
    const auto& axes = (*outAxes)[i];
    if (axes.ndim() <= 0 || isIdentity(axes))
      continue;
    for (const auto& edge : node->outputs[i]) {
      NodeInfo newInfo(kFWD, axes, true);
      auto ndPtr = insertTransposeBetween(&edge, newInfo);
      created.insert(ndPtr);
    }
  }
  return created;
}


}  // namespace exec
}  // namespace mxnet
