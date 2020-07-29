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
 *  Copyright (c) 2016 by Contributors
 * \file c_api_executor.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api_cac.h>
#include <mxnet/executor.h>

#include "./c_api_common.h"
#include "../executor/graph_executor.h"

int MXExecutorBackwardExWithGradientSkip(ExecutorHandle handle,
                                         mx_uint len,
                                         NDArrayHandle *head_grads,
                                         int is_train,
                                         GradientSkipInfo *gs_info,
                                         mx_uint gs_info_len,
                                         int gs_stop_layer_num,
                                         int gs_non_stop_layer_num,
                                         int *gs_stop_layer_num_update) {
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  std::vector<NDArray> ndarrays;
  NDArray **args_ptr = reinterpret_cast<NDArray**>(head_grads);
  for (mx_uint i = 0; i < len; ++i) {
    ndarrays.push_back(*args_ptr[i]);
  }
  exec->Backward(ndarrays, is_train, gs_info, gs_info_len, gs_stop_layer_num, gs_non_stop_layer_num,
                 gs_stop_layer_num_update);
  API_END();
}
