// Copyright 2020 Google LLC
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

// Evaluates a directory of models against the target model, as part of the rl
// loop. It uses the single-model evaluation class set up in cc/eval.h and
// cc/eval.cc.

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_ML_PERF_EVAL_MODELS_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_ML_PERF_EVAL_MODELS_H_

#include "REDACTEDflags/flag.h"

ABSL_DECLARE_FLAG(int32_t, start);
ABSL_DECLARE_FLAG(int32_t, end);
ABSL_DECLARE_FLAG(std::string, timestamp_file);
ABSL_DECLARE_FLAG(int32_t, num_games);
ABSL_DECLARE_FLAG(std::string, start_file);
ABSL_DECLARE_FLAG(std::string, model_dir);
ABSL_DECLARE_FLAG(double, target_winrate);
ABSL_DECLARE_FLAG(std::vector<std::string>, devices);

namespace minigo {

struct ModelInfoTuple {
  std::string timestamp, name, model_path;
};

std::vector<ModelInfoTuple> load_train_times();

void EvaluateModels(std::vector<minigo::ModelInfoTuple> models, int num_games);

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_ML_PERF_EVAL_MODELS_H_
