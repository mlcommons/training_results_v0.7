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

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_ASYNC_POLL_THREAD_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_ASYNC_POLL_THREAD_H_

#include <atomic>
#include <functional>
#include <string>

#include "REDACTEDsynchronization/mutex.h"
#include "REDACTEDtime/time.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread.h"

namespace minigo {

// Poller starts a thread that calls its abstract `Poll` method at a regular
// interval.
class PollThread : public Thread {
 public:
  PollThread(std::string thread_name, absl::Duration poll_interval,
             std::function<void()> poll_fn);

  virtual ~PollThread();

  void Join() override;

 private:
  void Run() override LOCKS_EXCLUDED(&mutex_);
  bool IsJoining() const EXCLUSIVE_LOCKS_REQUIRED(&mutex_);

  const absl::Duration poll_interval_;

  absl::Mutex mutex_;
  bool is_joining_ GUARDED_BY(&mutex_) = false;
  std::function<void()> poll_fn_ GUARDED_BY(&mutex_);
};

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_ASYNC_POLL_THREAD_H_
