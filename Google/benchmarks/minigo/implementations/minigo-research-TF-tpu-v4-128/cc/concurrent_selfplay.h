#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_CONCURRENT_SELFPLAY_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_CONCURRENT_SELFPLAY_H_

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

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "base/commandlineflags.h"
#include "REDACTEDmemory/memory.h"
#include "REDACTEDstrings/str_cat.h"
#include "REDACTEDstrings/str_join.h"
#include "REDACTEDstrings/str_replace.h"
#include "REDACTEDstrings/str_split.h"
#include "REDACTEDsynchronization/mutex.h"
#include "REDACTEDtime/clock.h"
#include "REDACTEDtime/time.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/poll_thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/sharded_executor.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread_safe_queue.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/directory_watcher.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/path.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/game.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/game_utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/init.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/mcts_tree.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/inference_cache.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/model/loader.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/platform/utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/random.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/tf_utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/wtf_saver.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/zobrist.h"
#include "third_party/tensorflow/core/lib/io/path.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/file_system.h"
#include "third_party/tracing_framework_bindings_cpp/macros.h"

// Plays multiple selfplay games.
// There are several important classes in this binary:
//  - `SelfplayGame` : holds the state for a single game, most importantly an
//    `MctsTree` and a `Game`. The `SelfplayGame` is responsible for selecting
//    leaves in the MCTS tree to run inference on, propagating inference
//    results back up the tree, and playing moves.
//  - `SelfplayThread` : owns multiple `SelfplayGame` instances and uses them
//    to play games concurrently. See SelfplayThread::Run for the sequence of
//    operations performed when playing games. Tree search is carried out in
//    batches on multiple threads in parallel.
//  - `Selfplayer` : owns multiple `SelfplayThread` instances, which lets the
//    binary perform tree search on multiple threads.
//  - `OutputThread` : responsible for writing SGF & training examples to
//    storage. After a game finished, its `SelfplayThread` hands the
//    `SelfplayGame` instance back to the `Selfplayer`, which pushes it onto
//    an output queue for `OutputThread` to consume.

// Inference flags.
DECLARE_string(engine);
DECLARE_string(device);
DECLARE_string(model);
DECLARE_int32(cache_size_mb);
DECLARE_int32(cache_shards);

// Tree search flags.
DECLARE_int32(num_readouts);
DECLARE_double(fastplay_frequency);
DECLARE_int32(fastplay_readouts);
DECLARE_int32(virtual_losses);
DECLARE_double(dirichlet_alpha);
DECLARE_double(noise_mix);
DECLARE_double(value_init_penalty);
DECLARE_bool(target_pruning);
DECLARE_double(policy_softmax_temp);
DECLARE_bool(allow_pass);
DECLARE_int32(restrict_pass_alive_play_threshold);

// Threading flags.
DECLARE_int32(selfplay_threads);
DECLARE_int32(parallel_search);
DECLARE_int32(parallel_inference);
DECLARE_int32(concurrent_games_per_thread);

// Game flags.
DECLARE_uint64(seed);
DECLARE_double(min_resign_threshold);
DECLARE_double(max_resign_threshold);
DECLARE_double(disable_resign_pct);
DECLARE_int32(num_games);
DECLARE_bool(run_forever);
DECLARE_string(abort_file);

// Output flags.
DECLARE_double(holdout_pct);
DECLARE_string(output_dir);
DECLARE_string(holdout_dir);
DECLARE_string(sgf_dir);
DECLARE_string(wtf_trace);
DECLARE_bool(verbose);
DECLARE_int32(output_threads);

namespace minigo {

inline std::string GetOutputDir(absl::Time now, const std::string& model_name,
                                const std::string& root_dir) {
  auto sub_dirs = absl::FormatTime("%Y-%m-%d-%H", now, absl::UTCTimeZone());
  auto clean_model_name =
      absl::StrReplaceAll(model_name, {{":", "_"}, {"/", "_"}, {".", "_"}});
  std::string processed_root_dir =
      absl::StrReplaceAll(root_dir, {{"$MODEL", clean_model_name}});
  return file::JoinPath(processed_root_dir, sub_dirs);
}

// Information required to run a single inference.
struct Inference {
  InferenceCache::Key cache_key;
  MctsNode* leaf;
  ModelInput input;
  ModelOutput output;
};

// Holds all the state for a single selfplay game.
// Each `SelfplayThread` plays multiple games in parallel, calling
// `SelectLeaves`, `ProcessInferences` and `MaybePlayMove` sequentially.
class SelfplayGame {
 public:
  struct Options {
    // Number of virtual losses.
    int num_virtual_losses;

    // Number of positions to read normally.
    int num_readouts;

    // Number of positions to read if playout cap oscillations determines that
    // this should be a "fast" play.
    int fastplay_readouts;

    // Frequency that a move should be a "fast" play.
    float fastplay_frequency;

    // Alpha value for Dirichlet noise.
    float dirichlet_alpha;

    // Fraction of noise to mix into the root node before performing reads.
    // Noise is not injected for "fast" plays.
    float noise_mix;

    // True if this game's data should be written to the `holdout_dir` instead
    // of the `output_dir`.
    bool is_holdout;

    // If true, subtract visits from all moves that weren't the best move until
    // the uncertainty level compensates.
    bool target_pruning;

    // If true, perform verbose logging. Usually restricted to just the first
    // `SelfplayGame` of the first `SelfplayThread`.
    bool verbose;

    // If false, pass is only read and played if there are no other legal
    // alternatives.
    bool allow_pass;

    // Disallow playing in pass-alive territory once the number of passes played
    // during a game is at least `restrict_pass_alive_play_threshold`.
    int restrict_pass_alive_play_threshold;
  };

  // Stats about the nodes visited during SelectLeaves.
  struct SelectLeavesStats {
    int num_leaves_queued = 0;
    int num_nodes_selected = 0;
    int num_cache_hits = 0;
    int num_game_over_leaves = 0;

    SelectLeavesStats& operator+=(const SelectLeavesStats& other) {
      num_leaves_queued += other.num_leaves_queued;
      num_nodes_selected += other.num_nodes_selected;
      num_cache_hits += other.num_cache_hits;
      num_game_over_leaves += other.num_game_over_leaves;
      return *this;
    }
  };

  SelfplayGame(int game_id, const Options& options, std::unique_ptr<Game> game,
               std::unique_ptr<MctsTree> tree);

  int game_id() const { return game_id_; }
  Game* game() { return game_.get(); }
  const Game* game() const { return game_.get(); }
  const MctsTree* tree() const { return tree_.get(); }
  absl::Duration duration() const { return duration_; }
  const Options& options() const { return options_; }
  const std::vector<std::string>& models_used() const { return models_used_; }

  // Selects leaves to perform inference on.
  // Returns the number of leaves selected. It is possible that no leaves will
  // be selected if all desired leaves are already in the inference cache.
  SelectLeavesStats SelectLeaves(InferenceCache* cache,
                                 std::vector<Inference>* inferences);

  // Processes the inferences selected by `SelectedLeaves` that were evaluated
  // by the SelfplayThread.
  void ProcessInferences(const std::string& model_name,
                         absl::Span<const Inference> inferences);

  // Plays a move if the necessary number of nodes have been read.
  // Returns true if `MaybePlayMove` actually played a move.
  // Returns false if the `SeflplayGame` needs to read more positions before it
  // can play a move.
  bool MaybePlayMove();

 private:
  // Randomly choose whether or not to fast play.
  bool ShouldFastplay();

  // Returns true if the predicted win rate is below `resign_threshold`.
  bool ShouldResign() const;

  // Injects noise into the root.
  void InjectNoise();

  // Returns the symmetry that should be when performing inference on this
  // node's position.
  symmetry::Symmetry GetInferenceSymmetry(const MctsNode* node) const;

  // Looks the `leaf` up in the inference cache:
  //  - if found: propagates the cached inference result back up the tree.
  //  - if not found: appends an element to `inferences` to perform inference
  //    on `leaf`.
  // Returns true in an inference was queued.
  bool MaybeQueueInference(MctsNode* leaf, InferenceCache* cache,
                           std::vector<Inference>* inferences);

  const Options options_;
  int target_readouts_;
  std::unique_ptr<Game> game_;
  std::unique_ptr<MctsTree> tree_;
  const bool use_ansi_colors_;
  const absl::Time start_time_;
  absl::Duration duration_;
  std::vector<std::string> models_used_;
  Random rnd_;
  const uint64_t inference_symmetry_mix_;

  // We need to wait until the root is expanded by the first call to
  // SelectLeaves in the game before injecting noise.
  bool inject_noise_before_next_read_ = false;

  // We don't allow fast play for the opening move: fast play relies to some
  // degree on tree reuse from earlier reads but the tree is empty at the start
  // of the game.
  bool fastplay_ = false;

  // Number of consecutive passes played by black and white respectively.
  // Used to determine when to disallow playing in pass-alive territory.
  // `num_consecutive_passes_` latches once it reaches
  // `restrict_pass_alive_play_threshold` is is not reset to 0 when a non-pass
  // move is played.
  int num_consecutive_passes_[2] = {0, 0};

  const int game_id_;
};

// The main application class.
// Manages multiple SelfplayThread objects.
// Each SelfplayThread plays multiple games concurrently, each one is
// represented by a SelfplayGame.
// The Selfplayer also has a OutputThread, which writes the results of completed
// games to disk.
class Selfplayer {
 public:
  Selfplayer();

  void Run() LOCKS_EXCLUDED(&mutex_);

  std::unique_ptr<SelfplayGame> StartNewGame(bool verbose)
      LOCKS_EXCLUDED(&mutex_);

  void EndGame(std::unique_ptr<SelfplayGame> selfplay_game)
      LOCKS_EXCLUDED(&mutex_);

  // Exectutes `fn` on `parallel_search` threads in parallel on a shared
  // `ShardedExecutor`.
  // Concurrent calls to `ExecuteSharded` are executed sequentially, unless
  // `parallel_search == 1`. This blocking property can be used to pipeline
  // CPU tree search and GPU inference.
  void ExecuteSharded(std::function<void(int, int)> fn);

  // Grabs a model from a pool. If `selfplay_threads > parallel_inference`,
  // `AcquireModel` may block if a model isn't immediately available.
  std::unique_ptr<Model> AcquireModel();

  // Gives a previously acquired model back to the pool.
  void ReleaseModel(std::unique_ptr<Model> model);

 private:
  void ParseFlags() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);
  FeatureDescriptor InitializeModels();
  void CreateModels(const std::string& path);
  void CheckAbortFile();

  mutable absl::Mutex mutex_;
  MctsTree::Options tree_options_ GUARDED_BY(&mutex_);
  int num_games_remaining_ GUARDED_BY(&mutex_) = 0;
  Random rnd_ GUARDED_BY(&mutex_);
  WinStats win_stats_ GUARDED_BY(&mutex_);
  ThreadSafeQueue<std::unique_ptr<SelfplayGame>> output_queue_;
  ShardedExecutor executor_;

  ThreadSafeQueue<std::unique_ptr<Model>> models_;

  // The latest path that matches the model pattern.
  std::string latest_model_name_ GUARDED_BY(&mutex_);

  int next_game_id_ GUARDED_BY(&mutex_) = 1;

  std::unique_ptr<DirectoryWatcher> directory_watcher_;
  std::unique_ptr<PollThread> abort_file_watcher_;

  std::unique_ptr<WtfSaver> wtf_saver_;
};

// Plays multiple games concurrently using `SelfplayGame` instances.
class SelfplayThread : public Thread {
 public:
  SelfplayThread(int thread_id, Selfplayer* selfplayer,
                 std::shared_ptr<InferenceCache> cache);

 private:
  void Run() override;

  // Starts new games playing.
  void StartNewGames();

  // Selects leaves to perform inference on for all currently playing games.
  // The selected leaves are stored in `inferences_` and `inference_spans_`
  // maps contents of `inferences_` back to the `SelfplayGames` that they
  // came from.
  void SelectLeaves();

  // Runs inference on the leaves selected by `SelectLeaves`.
  // Runs the name of the model that ran the inferences.
  std::string RunInferences();

  // Calls `SelfplayGame::ProcessInferences` for all inferences performed.
  void ProcessInferences(const std::string& model);

  // Plays moves on all games that have performed sufficient reads.
  void PlayMoves();

  struct TreeSearch {
    // Holds the span of inferences requested for a single `SelfplayGame`:
    // `pos` and `len` index into the `inferences` array.
    struct InferenceSpan {
      SelfplayGame* selfplay_game;
      size_t pos;
      size_t len;
    };

    void Clear() {
      inferences.clear();
      inference_spans.clear();
    }

    std::vector<Inference> inferences;
    std::vector<InferenceSpan> inference_spans;
  };

  Selfplayer* selfplayer_;
  int num_virtual_losses_ = 8;
  std::vector<std::unique_ptr<SelfplayGame>> selfplay_games_;
  std::unique_ptr<Model> model_;
  std::shared_ptr<InferenceCache> cache_;
  std::vector<TreeSearch> searches_;
  int num_games_finished_ = 0;
  const int thread_id_;
};

// Writes SGFs and training examples for completed games to disk.
class OutputThread : public Thread {
 public:
  OutputThread(int thread_id, FeatureDescriptor feature_descriptor,
               ThreadSafeQueue<std::unique_ptr<SelfplayGame>>* output_queue);

 private:
  void Run() override;
  void WriteOutputs(std::unique_ptr<SelfplayGame> selfplay_game);

  ThreadSafeQueue<std::unique_ptr<SelfplayGame>>* output_queue_;
  const std::string output_dir_;
  const std::string holdout_dir_;
  const std::string sgf_dir_;
  const FeatureDescriptor feature_descriptor_;
};

}  // namespace minigo

#endif  // MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_CONCURRENT_SELFPLAY_H_
