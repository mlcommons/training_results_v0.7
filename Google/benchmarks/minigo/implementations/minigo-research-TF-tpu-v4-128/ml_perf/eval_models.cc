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

#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/ml_perf/eval_models.h"

#include <chrono>  // NOLINT
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <istream>
#include <regex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <tuple>

#include "strings/numbers.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/eval.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/directory_watcher.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/path.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/game_utils.h"
#include "third_party/tensorflow/core/lib/io/path.h"
#include "third_party/tensorflow/core/platform/env.h"

ABSL_FLAG(int32_t, start, 0,
          "Index of first model to evaluate. If not"
          "specified, evaluate every game");
ABSL_FLAG(int32_t, end, kint32max, "Index of last model to evaluate.");
ABSL_FLAG(std::string, timestamp_file, "",
          "A file records the model name and timestamp");
ABSL_FLAG(int32_t, num_games, 128, "Number of games to run.");
ABSL_FLAG(std::string, start_file, "",
          "File generated at the end of training loop to notify the start of "
          "the evaluation process. Same as the abort file for selfplay. If not"
          "specified, evaluation starts immediately");
ABSL_FLAG(std::string, model_dir, "", "Model directory.");
ABSL_FLAG(double, target_winrate, 0.5,
          "Fraction of games that a model must beat the target by.");
ABSL_FLAG(std::vector<std::string>, devices, {}, "List of devices to run on.");

namespace minigo {

const int start_time = absl::ToUnixSeconds(absl::Now());

std::vector<ModelInfoTuple> load_train_times() {
  std::vector<ModelInfoTuple> models;
  std::string path =
      tensorflow::io::JoinPath(absl::GetFlag(FLAGS_timestamp_file));
  LOG(INFO) << "About to read file " << path;
  std::string timestamp_data;
  const auto& status =
      ReadFileToString(tensorflow::Env::Default(), path, &timestamp_data);
  if (!status.ok()) {
    LOG(ERROR) << "Cannot open file " << path << std::endl;
  }
  std::stringstream line_stream(timestamp_data);
  std::string line;
  while (std::getline(line_stream, line, '\n')) {
    if (!line.empty()) {
      std::regex pattern("(\\s*)(\\S+)(\\s+)(\\S+)");
      std::smatch match;
      std::regex_search(line, match, pattern);
      std::string timestamp = match[2];
      std::string name = match[4];
      std::string model_path;
      model_path =
          tensorflow::io::JoinPath(absl::GetFlag(FLAGS_model_dir), name);
      struct ModelInfoTuple model_info = {timestamp, name, model_path};
      models.push_back(model_info);
    }
  }
  return models;
}

void EvaluateModels(std::vector<minigo::ModelInfoTuple> models, int num_games) {
  int start_point = 0;
  int num_model = models.size();
  while (stoi(models[start_point].name) < absl::GetFlag(FLAGS_start)) {
    start_point += 1;
    if (start_point >= num_model) {
      LOG(ERROR) << "Start point left no model to evaluate.";
      return;
    }
  }
  std::vector<std::string> devices = absl::GetFlag(FLAGS_devices);
  int num_devices = devices.size();

  std::string eval_model_name;
  std::string target_model_name;

  std::string target_path = absl::GetFlag(FLAGS_target_model);
  std::vector<std::unique_ptr<Evaluator>> evaluators_;
  evaluators_.reserve(num_devices);
  for (int i = 0; i < num_devices; i++) {
    evaluators_.push_back(absl::MakeUnique<Evaluator>());
  }
  bool reset = false;
  double time_value;
  for (int j = start_point;
       j < num_model && stoi(models[j].name) <= absl::GetFlag(FLAGS_end); j++) {
    LOG(INFO) << ":::MLL " << absl::ToUnixSeconds(absl::Now())
              << " eval_start: {\"value\": null, \"metadata\": "
              << "{\"epoch_num\": " << stoi(models[j].name) << ", "
              << "\"lineno\": 111, "
              << "\"file\": \"minigo/ml_perf/eval_models.cc\"}}";
    int total_wins = 0;
    int total_num_games = 0;
    for (int i = 0; i < num_devices; i++) {
      int low_end = i * num_games / num_devices;
      int high_end = (i + 1) * num_games / num_devices;
      int num_games_to_calculate = high_end - low_end;

      absl::SetFlag(&FLAGS_eval_model, models[j].model_path);
      absl::SetFlag(&FLAGS_parallel_games, num_games_to_calculate);
      absl::SetFlag(&FLAGS_eval_device, devices[i]);
      absl::SetFlag(&FLAGS_target_device, devices[i]);
      if (reset) {
        evaluators_[i]->Reset();
      } else {
        reset = true;
      }
      std::vector<std::pair<std::string, WinStats>> win_stats_result =
          evaluators_[i]->Run();
      WinStats eval_stats = win_stats_result[0].second;
      WinStats target_stats = win_stats_result[1].second;
      eval_model_name = win_stats_result[0].first;
      target_model_name = win_stats_result[1].first;
      int num_games_i =
          eval_stats.black_wins.total() + eval_stats.white_wins.total() +
          target_stats.black_wins.total() + target_stats.white_wins.total();
      total_wins +=
          eval_stats.black_wins.total() + eval_stats.white_wins.total();
      total_num_games += num_games_i;
    }
    double win_rate = (double)total_wins / (double)total_num_games;
    LOG(INFO) << ":::MLL " << absl::ToUnixSeconds(absl::Now())
              << " eval_stop: {\"value\": null, \"metadata\": "
              << "{\"epoch_num\": " << stoi(models[j].name) << ", "
              << " \"lineno\": 149, "
              << "\"file\": \"minigo/ml_perf/eval_models.cc\"}}";
    if (!strings::safe_strtod(models[j].timestamp, &time_value)) {
      LOG(ERROR) << "Could not convert time_value to float";
    }
    LOG(INFO) << ":::MLL " << int(start_time + time_value)
              << " eval_accuracy: {\"value\": " << win_rate
              << ", \"metadata\": "
              << "{\"epoch_num\": " << stoi(models[j].name) << ", "
              << " \"lineno\": 158, "
              << "\"file\": \"minigo/ml_perf/eval_models.cc\"}}";

    LOG(INFO) << "Win rate " << eval_model_name << " vs " << target_model_name
              << " : " << std::fixed << std::setprecision(3) << win_rate;
    if (win_rate >= absl::GetFlag(FLAGS_target_winrate)) {
      LOG(INFO) << "Model " << models[j].name << " beat target after "
                << models[j].timestamp << " s.";
      LOG(INFO) << ":::MLL " << int(start_time + time_value)
                << " run_stop: {\"value\": null, \"metadata\": "
                << "{\"status\": \"success\","
                << " \"lineno\": 173, "
                << "\"file\": \"minigo/ml_perf/eval_models.cc\"}}";
      return;
    }
  }
  double last_time_value;
  if (!strings::safe_strtod(models.back().timestamp, &last_time_value)) {
    LOG(ERROR) << "Could not convert time_value to float";
  }
  LOG(INFO) << ":::MLL " << int(start_time + last_time_value)
            << " run_stop: {\"value\": null, \"metadata\": "
            << "{\"status\": \"failure\","
            << " \"lineno\": 185, "
            << "\"file\": \"minigo/ml_perf/eval_models.cc\"}}";
}
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(absl::GetFlag(FLAGS_seed));
  QCHECK(!absl::GetFlag(FLAGS_target_model).empty())
      << "FLAGS_target_model not specified! ";
  QCHECK(!absl::GetFlag(FLAGS_timestamp_file).empty())
      << "FLAGS_target_model not specified! ";

  if (!absl::GetFlag(FLAGS_target_model).empty()) {
    while (!minigo::file::FileExists(absl::GetFlag(FLAGS_start_file))) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  std::vector<minigo::ModelInfoTuple> models = minigo::load_train_times();
  minigo::EvaluateModels(models, absl::GetFlag(FLAGS_num_games));
}
