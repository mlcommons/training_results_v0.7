// Copyright 2019 Google LLC
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

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_WTF_SAVER_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_WTF_SAVER_H_

#include <string>

#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread.h"
#include "third_party/tracing_framework_bindings_cpp/runtime.h"

namespace minigo {

class WtfSaver {
 public:
  explicit WtfSaver(std::string path, absl::Duration poll_interval);
  ~WtfSaver();

 private:
  void Poll();

  const std::string path_;
  wtf::Runtime::SaveOptions options_;
  wtf::Runtime::SaveCheckpoint checkpoint_;
  std::unique_ptr<Thread> poll_thread_;
};

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_WTF_SAVER_H_
