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
 *  Copyright (c) 2015 by Contributors
 * \file c_api.h
 * \brief C API of mxnet
 */
#ifndef MXNET_C_API_CAC_H_
#define MXNET_C_API_CAC_H_

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <mxnet/c_api.h>

struct GradientSkipInfo {
  char *key;
  int skip_term;
  int layer_num;
};

/*!
 * \brief Excecutor run backward
 *
 * \param handle execute handle
 * \param len lenth
 * \param head_grads NDArray handle for heads' gradient
 * \param is_train int value to indicate whether the backward pass is for evaluation
 *
 * \return 0 when success, -1 when failure happens
 */
MXNET_DLL int MXExecutorBackwardExWithGradientSkip(ExecutorHandle handle,
                                                   mx_uint len,
                                                   NDArrayHandle *head_grads,
                                                   int is_train,
                                                   GradientSkipInfo *gs_info = nullptr,
                                                   mx_uint gs_info_len = 0,
                                                   int gs_stop_layer_num = -1,
                                                   int gs_non_stop_layer_num = -1,
                                                   int *gs_stop_layer_num_update = nullptr);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // MXNET_C_API_CAC_H_
