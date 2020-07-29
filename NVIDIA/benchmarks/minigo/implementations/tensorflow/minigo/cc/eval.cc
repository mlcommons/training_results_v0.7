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

#include <stdio.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_player.h"
#include "cc/model/batching_model.h"
#include "cc/model/loader.h"
#include "cc/model/model.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/zobrist.h"

namespace minigo {
namespace {

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
  Evaluator(bool   resign_enabled,
            double resign_threshold,
            uint64_t seed,
            int32_t virtual_losses,
            double value_init_penalty,
            std::string eval_model,
            std::string eval_device,
            int32_t num_eval_readouts,
            std::string target_model,
            std::string target_device,
            int32_t num_target_readouts,
            int32_t parallel_games,
            bool verbose) 

          : nv_resign_enabled(resign_enabled),
            nv_resign_threshold(resign_threshold),
            nv_seed(seed),
            nv_virtual_losses(virtual_losses),
            nv_value_init_penalty(value_init_penalty),
            nv_eval_model(eval_model),
            nv_eval_device(eval_device),
            nv_num_eval_readouts(num_eval_readouts),
            nv_target_model(target_model),
            nv_target_device(target_device),
            nv_num_target_readouts(num_target_readouts),
            nv_parallel_games(parallel_games),
            nv_verbose(verbose) {

    // Create a batcher for the eval model.
    batchers_.push_back(
        absl::make_unique<BatchingModelFactory>(nv_eval_device, 2));

    // If the target model requires a different device, create one & a second
    // batcher too.
    if (nv_target_device != nv_eval_device) {
      batchers_.push_back(
          absl::make_unique<BatchingModelFactory>(nv_target_device, 2));
    }
  }

  double Run() {
    auto start_time = absl::Now();

    game_options_.resign_enabled = nv_resign_enabled;
    game_options_.resign_threshold = -std::abs(nv_resign_threshold);

    MctsPlayer::Options player_options;
    player_options.virtual_losses = nv_virtual_losses;
    player_options.inject_noise = false;
    player_options.random_seed = nv_seed;
    player_options.tree.value_init_penalty = nv_value_init_penalty;
    player_options.tree.soft_pick_enabled = false;

    player_options.num_readouts = nv_num_eval_readouts;
    EvaluatedModel eval_model(batchers_.front().get(), nv_eval_model, player_options);

    player_options.num_readouts = nv_num_target_readouts;
    EvaluatedModel target_model(batchers_.back().get(), nv_target_model, player_options);

    int num_games = nv_parallel_games;
    for (int thread_id = 0; thread_id < num_games; ++thread_id) {
      bool swap_models = (thread_id & 1) != 0;
      threads_.emplace_back(
          std::bind(&Evaluator::ThreadRun, this, thread_id,
                    swap_models ? &target_model : &eval_model,
                    swap_models ? &eval_model : &target_model)); 
    }
    for (auto& t : threads_) {
      t.join();
    }

    if(nv_verbose) {
        MG_LOG(INFO) << "Evaluated " << num_games << " games, total time "
                     << (absl::Now() - start_time);
        MG_LOG(INFO) << FormatWinStatsTable(
            {{eval_model.name(), eval_model.GetWinStats()},
             {target_model.name(), target_model.GetWinStats()}});
    }

    uint64_t eval_model_wins = eval_model.GetWinStats().black_wins.total() + eval_model.GetWinStats().white_wins.total();
    uint64_t target_model_wins = target_model.GetWinStats().black_wins.total() + target_model.GetWinStats().white_wins.total();
    double win_rate = double(eval_model_wins) / (eval_model_wins + target_model_wins);
    return win_rate;
  }

 private:
  void ThreadRun(int thread_id, EvaluatedModel* black_model,
                 EvaluatedModel* white_model) {
    // Only print the board using ANSI colors if stderr is sent to the
    // terminal.
    const bool use_ansi_colors = FdSupportsAnsiColors(fileno(stderr));

    // The player and other_player reference this pointer.
    std::unique_ptr<Model> model;
    Game game(black_model->name(), white_model->name(), game_options_);

    const bool verbose = nv_verbose && (thread_id == 0);
    auto black = absl::make_unique<MctsPlayer>(
        black_model->NewModel(), nullptr, &game, black_model->player_options());
    auto white = absl::make_unique<MctsPlayer>(
        white_model->NewModel(), nullptr, &game, white_model->player_options());

    BatchingModelFactory::StartGame(black->model(), white->model());
    auto* curr_player = black.get();
    auto* next_player = white.get();
    while (!game.game_over()) {
      if (curr_player->root()->position.n() >= kMinPassAliveMoves &&
          curr_player->root()->position.CalculateWholeBoardPassAlive()) {
        // Play pass moves to end the game.
        while (!game.game_over()) {
          MG_CHECK(curr_player->PlayMove(Coord::kPass));
          next_player->PlayOpponentsMove(Coord::kPass);
          std::swap(curr_player, next_player);
        }
        break;
      }

      auto move = curr_player->SuggestMove(curr_player->options().num_readouts);
      if (verbose) {
        std::cerr << curr_player->tree().Describe() << "\n";
      }
      MG_CHECK(curr_player->PlayMove(move));
      if (move != Coord::kResign) {
        next_player->PlayOpponentsMove(move);
      }
      if (verbose) {
        MG_LOG(INFO) << absl::StreamFormat(
            "%d: %s by %s\nQ: %0.4f", curr_player->root()->position.n(),
            move.ToGtp(), curr_player->name(), curr_player->root()->Q());
        MG_LOG(INFO) << curr_player->root()->position.ToPrettyString(
            use_ansi_colors);
      }
      std::swap(curr_player, next_player);
    }
    BatchingModelFactory::EndGame(black->model(), white->model());

    if (game.result() > 0) {
      black_model->UpdateWinStats(game);
    } else {
      white_model->UpdateWinStats(game);
    }

    if (verbose) {
      MG_LOG(INFO) << game.result_string();
      MG_LOG(INFO) << "Black was: " << game.black_name();

      MG_LOG(INFO) << "Thread " << thread_id << " stopping";
    }

  }

  Game::Options game_options_;
  std::vector<std::thread> threads_;
  std::atomic<size_t> game_id_{0};
  std::vector<std::unique_ptr<BatchingModelFactory>> batchers_;

  //nv_args that were flags before
  bool   nv_resign_enabled;
  double nv_resign_threshold;
  uint64_t nv_seed;
  int32_t nv_virtual_losses;
  double nv_value_init_penalty;
  std::string nv_eval_model;
  std::string nv_eval_device;
  int32_t nv_num_eval_readouts;
  std::string nv_target_model;
  std::string nv_target_device;
  int32_t nv_num_target_readouts;
  int32_t nv_parallel_games;
  bool nv_verbose; 

};

}  // namespace

namespace eval {

double run(bool   resign_enabled,
           double resign_threshold,
           uint64_t seed,
           int32_t virtual_losses,
           double value_init_penalty,
           std::string eval_model,
           std::string eval_device,
           int32_t num_eval_readouts,
           std::string target_model,
           std::string target_device,
           int32_t num_target_readouts,
           int32_t parallel_games,
           bool verbose) {

  minigo::zobrist::Init(seed);
  minigo::Evaluator evaluator(resign_enabled,
                              resign_threshold,
                              seed,
                              virtual_losses,
                              value_init_penalty,
                              eval_model,
                              eval_device,
                              num_eval_readouts,
                              target_model,
                              target_device,
                              num_target_readouts,
                              parallel_games,
                              verbose);

  return evaluator.Run();
}

}  // namespace eval
}  // namespace minigo
