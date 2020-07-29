// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/dual_net/tpu_dual_net.h"

#include <algorithm>
#include <cstddef>
#include <memory>

#include "REDACTEDmemory/memory.h"
#include "REDACTEDstrings/match.h"
#include "REDACTEDstrings/numbers.h"
#include "REDACTEDstrings/str_cat.h"
#include "REDACTEDstrings/string_view.h"
#include "REDACTEDstrings/strip.h"
#include "REDACTEDtypes/span.h"
#include "REDACTEDinference/runner/options.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/constants.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/path.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "third_party/tensorflow/cc/saved_model/signature_constants.h"
#include "third_party/tensorflow/cc/saved_model/tag_constants.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/types.proto.h"
#include "third_party/tensorflow/core/lib/core/errors.h"
#include "third_party/tensorflow/core/lib/core/status.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/logging.h"
#include "third_party/tensorflow/core/platform/protobuf.h"
#include "third_party/tensorflow/core/protobuf/config.proto.h"
#include "third_party/tensorflow/core/public/session.h"
#include "third_party/tensorflow/core/public/session_options.h"
#include "third_party/tracing_framework_bindings_cpp/macros.h"

namespace minigo {

namespace {

// A GraphDef containing the ops required to initialize and shutdown a TPU.
// This proto was generated from the script oneoffs/generate_tpu_graph_def.py.
constexpr auto kTpuOpsGraphDef = R"(
node {
  name: "ConfigureDistributedTPU"
  op: "ConfigureDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
  attr {
    key: "embedding_config"
    value {
      s: ""
    }
  }
  attr {
    key: "is_global_init"
    value {
      b: false
    }
  }
  attr {
    key: "tpu_embedding_config"
    value {
      s: ""
    }
  }
}
node {
  name: "ShutdownDistributedTPU"
  op: "ShutdownDistributedTPU"
  device: "/device:TPU_SYSTEM:0"
}
library {
}
)";

std::unique_ptr<tensorflow::Session> CreateSession(
    const tensorflow::GraphDef& graph_def, const std::string& tpu_name) {
  // The following check is commented out since we are running the script on
  // REDACTED TPU instead of cloud TPU for now.
  // Make sure tpu_name is a gRPC address of cloud TPU.
  // MG_CHECK(absl::StartsWith(tpu_name, "grpc://"));

  tensorflow::SessionOptions options;
  options.target = tpu_name;
  options.config.set_allow_soft_placement(true);
  options.config.set_log_device_placement(true);
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(options));
  TF_CHECK_OK(session->Create(graph_def));
  return session;
}

}  // namespace

TpuDualNet::TpuDualNet(const std::string& graph_path,
                       const FeatureDescriptor& feature_desc,
                       const std::string& tpu_name)
    : Model(std::string(file::Stem(graph_path)), feature_desc),
      tpu_name_(tpu_name),
      graph_path_(graph_path),
      feature_descriptor_(feature_desc) {
  TF_CHECK_OK(GetModel());
}

TpuDualNet::~TpuDualNet() {
  absl::MutexLock lock(&mutex_);
  LOG(INFO) << "Shutting down TPU for TpuDualNet" << std::endl;
  saved_model_bundle_.session.reset();
}

void TpuDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                         std::vector<ModelOutput*>* outputs,
                         std::string* model_name) {
  auto input_size = static_cast<int>(inputs.size());
  MG_CHECK(input_size > 0);
  switch (feature_descriptor_.layout) {
    case FeatureDescriptor::Layout::kNhwc:
      input_ =
          tensorflow::Tensor(input_type_, tensorflow::TensorShape({
                                              static_cast<int>(input_size),
                                              kN,
                                              kN,
                                              feature_descriptor().num_planes,
                                          }));
      break;
    case FeatureDescriptor::Layout::kNchw:
      input_ =
          tensorflow::Tensor(input_type_, tensorflow::TensorShape({
                                              static_cast<int>(input_size),
                                              feature_descriptor().num_planes,
                                              kN,
                                              kN,
                                          }));
      break;
  }
  batch_capacity_ = input_size;
  WTF_SCOPE("TpuDualNet::Run: inputs, capacity", size_t, size_t)
  (input_size, batch_capacity_);

  {
    WTF_SCOPE("SetFeatures: inputs", size_t)(input_size);

    if (input_type_ == tensorflow::DT_FLOAT) {
      Tensor<float> features(feature_descriptor_.GetInputShape(input_size),
                             input_.flat<float>().data());
      feature_descriptor().set_floats(inputs, &features);
    } else {
      static_assert(sizeof(bool) == sizeof(uint8_t), "bool must be 1 byte");
      Tensor<uint8_t> features(
          feature_descriptor_.GetInputShape(input_size),
          reinterpret_cast<uint8_t*>(input_.flat<bool>().data()));
      feature_descriptor().set_bytes(inputs, &features);
    }
  }

  // Run the model.
  {
    WTF_SCOPE("Session::Run: inputs, capacity", size_t, size_t)
    (inputs.size(), batch_capacity_);
    run_outputs_.clear();
    TF_CHECK_OK(session_->Run({{input_tensor_names_, input_}},
                              output_tensor_names_, {}, &run_outputs_));
  }

  // Copy the policy and value out of the output tensors.
  {
    WTF_SCOPE("GetOutputs: outputs", size_t)(run_outputs_.size());
    const auto& tensor_0 = run_outputs_[0].flat<float>();
    const auto& tensor_1 = run_outputs_[1].flat<float>();
    Tensor<float> policy;
    Tensor<float> value;
    if (output_key_[0] == "policy_output") {
      policy = Tensor<float>({input_size, kNumMoves}, tensor_0.data());
      value = Tensor<float>({input_size}, tensor_1.data());
    } else {
      MG_CHECK(output_key_[1] == "policy_output");
      policy = Tensor<float>({input_size, kNumMoves}, tensor_1.data());
      value = Tensor<float>({input_size}, tensor_0.data());
    }
    Model::GetOutputs(inputs, policy, value, outputs);
  }

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

TpuDualNetFactory::TpuDualNetFactory(std::string tpu_name)
    : tpu_name_(std::move(tpu_name)) {
  // Create a session containing ops for initializing & shutting down a TPU.
  tensorflow::GraphDef graph_def;
  ::tensorflow::protobuf::TextFormat::ParseFromString(kTpuOpsGraphDef,
                                                      &graph_def);
  main_session_ = CreateSession(graph_def, tpu_name_);

  MG_LOG(INFO) << "Initializing TPU " << tpu_name_;
  TF_CHECK_OK(main_session_->Run({}, {}, {"ConfigureDistributedTPU"}, nullptr));
}

TpuDualNetFactory::~TpuDualNetFactory() {
  MG_LOG(INFO) << "Shutting down TPU " << tpu_name_;
  TF_CHECK_OK(main_session_->Run({}, {}, {"ShutdownDistributedTPU"}, nullptr));

  MG_LOG(INFO) << "Closing main session";
  TF_CHECK_OK(main_session_->Close());
}

tensorflow::Status TpuDualNet::GetModel() {
  absl::MutexLock lock(&mutex_);
  tensorflow::RunOptions run_options;
  std::unordered_set<std::string> tags = {tensorflow::kSavedModelTagServe,
                                          tensorflow::kSavedModelTagTpu};
  tensorflow::SessionOptions session_options;
  session_options.target = tpu_name_;
  session_options.config.set_allow_soft_placement(true);
  session_options.config.set_log_device_placement(true);

  TF_CHECK_OK(tensorflow::LoadSavedModel(
      session_options, run_options, graph_path_, tags, &saved_model_bundle_));

  // Get names of input and output tensors from signature.
  auto iter = saved_model_bundle_.meta_graph_def.signature_def().find(
      tensorflow::kDefaultServingSignatureDefKey);
  if (iter == saved_model_bundle_.meta_graph_def.signature_def().end()) {
    LOG(ERROR) << tensorflow::errors::InvalidArgument(
        absl::StrCat("Could not find SignatureDef with key: serving_default"));
  }
  signature_def_ = iter->second;
  for (const auto& input : signature_def_.inputs()) {
    input_tensor_names_ = input.second.name();
    input_type_ = input.second.dtype();
  }
  for (const auto& output : signature_def_.outputs()) {
    output_tensor_names_.push_back(output.second.name());
    output_key_.push_back(output.first);
  }
  session_ = std::move(saved_model_bundle_.session);
  return tensorflow::Status::OK();
}

std::unique_ptr<Model> TpuDualNetFactory::NewModel(const ModelDefinition& def) {
  // const std::string& descriptor) {
  MG_CHECK(def.metadata.Get<std::string>("engine") == "tpu");
  auto feature_desc =
      FeatureDescriptor::Create(def.metadata.Get<std::string>("input_features"),
                                def.metadata.Get<std::string>("input_layout"));
  // strip the .minigo at the end to use the savedmodel api
  std::string model_path = def.path;
  auto it = model_path.find(".minigo");
  if (it != std::string::npos) {
    model_path.erase(model_path.size() - 7);
  }
  return absl::make_unique<TpuDualNet>(model_path, feature_desc, tpu_name_);
}

}  // namespace minigo
