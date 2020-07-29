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

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_EVAL_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_EVAL_H_

#include <stdio.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "base/commandlineflags.h"
#include "REDACTEDmemory/memory.h"
#include "REDACTEDstrings/str_cat.h"
#include "REDACTEDstrings/str_format.h"
#include "REDACTEDstrings/str_split.h"
#include "REDACTEDtime/clock.h"
#include "REDACTEDtime/time.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/constants.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/path.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/game.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/game_utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/init.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/mcts_player.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/batching_model.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/loader.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/model.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/random.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/tf_utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/zobrist.h"

// Game options flags.
ABSL_DECLARE_FLAG(bool, resign_enabled);
ABSL_DECLARE_FLAG(double, resign_threshold);
ABSL_DECLARE_FLAG(uint64, seed);

// Tree search flags.
ABSL_DECLARE_FLAG(int32, virtual_losses);
ABSL_DECLARE_FLAG(double, value_init_penalty);

// Inference flags.
ABSL_DECLARE_FLAG(std::string, eval_model);
ABSL_DECLARE_FLAG(std::string, eval_device);
ABSL_DECLARE_FLAG(int32, num_eval_readouts);

ABSL_DECLARE_FLAG(std::string, target_model);
ABSL_DECLARE_FLAG(std::string, target_device);
ABSL_DECLARE_FLAG(int32, num_target_readouts);

ABSL_DECLARE_FLAG(int32, parallel_games);

// Output flags.
ABSL_DECLARE_FLAG(std::string, output_bigtable);
ABSL_DECLARE_FLAG(std::string, sgf_dir);
ABSL_DECLARE_FLAG(std::string, bigtable_tag);
ABSL_DECLARE_FLAG(bool, verbose);

namespace minigo {

class Evaluator {
  class EvaluatedModel {
   public:
    EvaluatedModel(BatchingModelFactory* batcher, const std::string& path,
                   const MctsPlayer::Options& player_options)
        : batcher_(batcher), path_(path), player_options_(player_options) {}

    std::string name() {
      absl::MutexLock lock(&mutex_);
      if (name_.empty()) {
        // The model's name is lazily initialized the first time we create a
        // instance. Make sure it's valid.
        NewModelImpl();
      }
      return name_;
    }

    WinStats GetWinStats() const {
      absl::MutexLock lock(&mutex_);
      return win_stats_;
    }

    void UpdateWinStats(const Game& game) {
      absl::MutexLock lock(&mutex_);
      win_stats_.Update(game);
    }

    std::unique_ptr<Model> NewModel() {
      absl::MutexLock lock(&mutex_);
      return NewModelImpl();
    }

    const MctsPlayer::Options& player_options() const {
      return player_options_;
    }

   private:
    std::unique_ptr<Model> NewModelImpl() EXCLUSIVE_LOCKS_REQUIRED(&mutex_) {
      auto model = batcher_->NewModel(path_);
      if (name_.empty()) {
        name_ = model->name();
      }
      return model;
    }

    mutable absl::Mutex mutex_;
    BatchingModelFactory* batcher_ GUARDED_BY(&mutex_);
    const std::string path_;
    std::string name_ GUARDED_BY(&mutex_);
    WinStats win_stats_ GUARDED_BY(&mutex_);
    MctsPlayer::Options player_options_;
  };

 public:
  Evaluator();

  void Reset();

  std::vector<std::pair<std::string, WinStats>> Run();

 private:
  void ThreadRun(int thread_id, EvaluatedModel* black_model,
                 EvaluatedModel* white_model);

  Game::Options game_options_;
  std::vector<std::thread> threads_;
  std::atomic<size_t> game_id_{0};
  std::vector<std::unique_ptr<BatchingModelFactory>> batchers_;
};

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_EVAL_H_
