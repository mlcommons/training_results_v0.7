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
 * \file cudnn_common_op.h
 * \brief
 * \author Dick Carter
*/

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_COMMON_OP_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_COMMON_OP_H_
#include <cudnn.h>
#include <dmlc/logging.h>
#include <vector>

namespace mxnet {
namespace op {
#if MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600

#if defined(__CUDACC__)
// A wrapper using RAII principles to simplify use of the cuDNN 'fused op' API.
class CuDNNCommonOp {
 public:
  explicit CuDNNCommonOp(cudnnFusedOps_t op_id) : plan_created_(false) {
    // New 'fused op' descriptor creation
    CUDNN_CALL(cudnnCreateFusedOpsPlan(&op_, op_id));
    CUDNN_CALL(cudnnCreateFusedOpsConstParamPack(&op_const_params_, op_id));
    CUDNN_CALL(cudnnCreateFusedOpsVariantParamPack(&op_variant_params_, op_id));
  }

  ~CuDNNCommonOp() {
    // New 'fused op' descriptor destruction
    CUDNN_CALL(cudnnDestroyFusedOpsVariantParamPack(op_variant_params_));
    CUDNN_CALL(cudnnDestroyFusedOpsConstParamPack(op_const_params_));
    CUDNN_CALL(cudnnDestroyFusedOpsPlan(op_));
  }

  // Launch op
  void Execute(cudnnHandle_t cudnn_handle) {
    CHECK(plan_created_) << "CuDNNCommonOp exec requested without a valid 'plan', need: "
                         << "<set const params>, GetWorkspaceSizeBytes(), Execute().";
    CUDNN_CALL(cudnnFusedOpsExecute(cudnn_handle, op_, op_variant_params_));
  }
  // Set a 'fused op' const param pack attribute given a descriptor (an opaque pointer) 'T'.
  template <typename T>
  void SetOpConstParamDesc(cudnnFusedOpsConstParamLabel_t param_label, T *param_ptr) {
    CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(op_const_params_, param_label, param_ptr));
    // Setting a 'const param pack' value invalidates the plan
    plan_created_ = false;
  }
  // Set multiple 'fused op' const param pack attribute given a descriptor (an opaque pointer) 'T'.
  template <typename T>
  void SetOpConstParamDesc(const std::vector<cudnnFusedOpsConstParamLabel_t> &param_labels,
                           T *param_ptr) {
    for (auto param_label : param_labels)
      SetOpConstParamDesc(param_label, param_ptr);
  }
  // Set a 'fused op' const param pack attribute given a value of param 'T'.
  template <typename T>
  void SetOpConstParamAttr(cudnnFusedOpsConstParamLabel_t param_label, T param) {
    CUDNN_CALL(cudnnSetFusedOpsConstParamPackAttribute(op_const_params_, param_label, &param));
    // Setting a 'const param pack' value invalidates the plan
    plan_created_ = false;
  }
  // Set multiple 'fused op' const param pack attributes given a value a param 'T'.
  template <typename T>
  void SetOpConstParamAttr(const std::vector<cudnnFusedOpsConstParamLabel_t> &param_labels,
                           T param) {
    for (auto param_label : param_labels)
      SetOpConstParamAttr(param_label, param);
  }
  // Set a 'fused op' variant param pack attribute given a reference to a param 'T'.
  template <typename T>
  void SetOpVariantParamAttrPtr(cudnnFusedOpsVariantParamLabel_t param_label, T *param_ptr) {
    CUDNN_CALL(cudnnSetFusedOpsVariantParamPackAttribute(op_variant_params_,
                                                         param_label, param_ptr));
  }
  // Set multiple 'fused op' const param pack attributes given a reference to a param 'T'.
  template <typename T>
  void SetOpVariantParamAttrPtr(const std::vector<cudnnFusedOpsVariantParamLabel_t> &param_labels,
                           const T *param_ptr) {
    for (auto param_label : param_labels)
      SetOpVariantParamAttrPtr(param_label, param_ptr);
  }
  // Get the workspace, which requires 'making a plan'.  This is required before Execute().
  size_t GetWorkspaceSizeInBytes(cudnnHandle_t cudnn_handle) {
    size_t workspace_bytes = 0U;
    CUDNN_CALL(cudnnMakeFusedOpsPlan(cudnn_handle, op_, op_const_params_, &workspace_bytes));
    plan_created_ = true;
    return workspace_bytes;
  }

 private:
  // `plan_created_` flag that helps diagnose an improper use. Need the sequence:
  // <set const params>
  // GetWorkspaceSizeInBytes() (a.k.a. 'make plan')
  // <set variant params>
  // Execute()
  bool plan_created_;

  // Op using the generalized 'FusedOp' API of cuDNN
  cudnnFusedOpsPlan_t op_;
  // Op parameters are held in a 'const parameter pack' of descriptor info and data ptr alignments.
  cudnnFusedOpsConstParamPack_t op_const_params_;
  // Op I/O data ptrs and 'light-weight' parameters are held in a 'variant param pack'
  cudnnFusedOpsVariantParamPack_t op_variant_params_;
};
#endif  // defined(__CUDACC__)

#endif  // MXNET_USE_CUDNN == 1 && CUDNN_VERSION >= 7600
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_COMMON_OP_H_
