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

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_DUAL_NET_TPU_DUAL_NET_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_DUAL_NET_TPU_DUAL_NET_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "REDACTEDcontainer/flat_hash_map.h"
#include "REDACTEDsynchronization/mutex.h"
#include "REDACTEDinference/runner/options.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/constants.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/factory.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/model.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/random.h"
#include "third_party/tensorflow/cc/saved_model/loader.h"
#include "third_party/tensorflow/core/framework/graph.pb.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/public/session.h"

namespace minigo {

// Model that runs inference on tpu.
class TpuDualNet : public Model {
 public:
  TpuDualNet(const std::string& graph_path,
             const FeatureDescriptor& feature_desc,
             const std::string& tpu_name);
  ~TpuDualNet() override;

  // Runs inference using TPU with the given inputs and fills the inference
  // outputs from policy and value tensors.
  // Args:
  //   inputs: a vector of ModelInput.
  //   outputs: a vector to be filled with outputs.
  //   model_name: path to the inference model's directory. THIS IS UNUSED
  //               since we have it when constructing TpuDualNet. It is here to
  //               override.
  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

  tensorflow::Status GetModel();

 private:
  tensorflow::Tensor input_;
  std::vector<tensorflow::Tensor> run_outputs_;
  size_t batch_capacity_ = 0;
  const std::string tpu_name_;
  const std::string graph_path_;
  tensorflow::SignatureDef signature_def_;
  tensorflow::DataType input_type_;
  std::string input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
  // output_key is a vector that contains "value_output" and "policy_output".
  // We use it to track the order output keys get pushed.
  std::vector<std::string> output_key_;
  tensorflow::SavedModelBundle saved_model_bundle_;
  absl::Mutex mutex_;
  std::unique_ptr<tensorflow::Session> session_;
  FeatureDescriptor feature_descriptor_;
};

// Factory that creates TpuDualNet instance.
class TpuDualNetFactory : public ModelFactory {
 public:
  TpuDualNetFactory(std::string tpu_name);
  ~TpuDualNetFactory() override;

  std::unique_ptr<Model> NewModel(const ModelDefinition& def) override;

 private:
  std::unique_ptr<tensorflow::Session> main_session_;
  std::string tpu_name_;
};

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_DUAL_NET_TPU_DUAL_NET_H_
