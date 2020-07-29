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
 * \file fuse_utils.cc
 * \brief
 * \author Clement Fuji Tsang
 */

#include "./exec_pass.h"

namespace mxnet {
namespace exec {

nnvm::NodeEntryMap<uint32_t> GetNodeEntryCount(const nnvm::Graph& g) {
  nnvm::NodeEntryMap<uint32_t> outputs;
  DFSVisit(g.outputs, [&outputs](const nnvm::NodePtr& node) {
    for (auto e : node->inputs) {
      outputs[e]++;
    }
  });
  for (auto e : g.outputs) {
    outputs[e]++;
  }
  return outputs;
}

Graph ReplaceNodeEntries(nnvm::Graph&& g,
                        const nnvm::NodeEntryMap<nnvm::NodeEntry>& entry_map) {
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    auto it = entry_map.find(g.outputs[i]);
    if (it != entry_map.end())
      g.outputs[i] = it->second;
  }
  DFSVisit(g.outputs, [&entry_map](const nnvm::NodePtr& n) {
    for (size_t i = 0; i < n->inputs.size(); ++i) {\
      auto it = entry_map.find(n->inputs[i]);
      if (it != entry_map.end())
        n->inputs[i] = it->second;
    }
  });
  return g;
}

}  // namespace exec
}  // namespace mxnet
