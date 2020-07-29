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

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_ASYNC_SEMAPHORE_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_ASYNC_SEMAPHORE_H_

#include "REDACTEDsynchronization/mutex.h"

namespace minigo {

class Semaphore {
 public:
  void Post() {
    absl::MutexLock lock(&mutex_);
    ++count_;
  }

  void Wait() {
    mutex_.LockWhen(absl::Condition(this, &Semaphore::is_non_zero));
    --count_;
    mutex_.Unlock();
  }

 private:
  bool is_non_zero() const EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
    return count_ != 0;
  }

  absl::Mutex mutex_;
  int count_ = 0;
};

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_ASYNC_SEMAPHORE_H_
