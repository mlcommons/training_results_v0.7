#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "REDACTEDflags/flag.h"
#include "REDACTEDflags/marshalling.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/poll_thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/init.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/sample_records.h"
#include "third_party/tensorflow/core/lib/io/path.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/file_system.h"

ABSL_FLAG(int32_t, num_games, 4096, "Number of games to wait for.");
ABSL_FLAG(int32_t, window_size, 5,
          "Maximum number of recent selfplay rounds to train on.");
ABSL_FLAG(int32_t, iterations, 43, "number of rl_loop iterations to run.");
ABSL_FLAG(std::string, golden_chunk_dir, "", "Training example directory.");
ABSL_FLAG(std::string, selfplay_dir, "", "Path to selfplay_dir.");
ABSL_FLAG(double, train_filter, 0.3,
          "Fraction of selfplay games to pass to training.");
namespace minigo {

namespace fs = std::filesystem;
std::vector<std::string> ListSelfplayDirs(std::string path) {
  std::vector<std::string> matched_entries;
  TF_CHECK_OK(tensorflow::Env::Default()->GetChildren(path, &matched_entries));
  std::sort(matched_entries.begin(), matched_entries.end(),
            std::greater<std::string>());
  return matched_entries;
}

void WaitForTrainingExamples(int model_num, int num_games,
                             std::string selfplay_dir) {
  bool first_time_around = true;
  std::string selfplay_model_name = absl::StrCat(
      std::string(6 - std::to_string(model_num).length(), '0'), model_num);
  fs::path selfplay_model_dir = fs::path(selfplay_dir) / selfplay_model_name;
  fs::path pattern = selfplay_model_dir / "*/*/*.tfrecord.zz";
  std::vector<std::string> matched_output_files;
  int counter = 0;
  while (true) {
    counter += 1;
    if (first_time_around) {
      LOG(INFO) << "Waiting for " << num_games << " games in "
                << selfplay_model_dir;
      LOG(INFO) << "Pattern as: " << pattern;
      first_time_around = false;
    }
    auto* env = tensorflow::Env::Default();
    if (env->FileExists(selfplay_model_dir).ok()) {
      TF_CHECK_OK(tensorflow::Env::Default()->GetMatchingPaths(
          pattern, &matched_output_files));
      if (matched_output_files.size() >= absl::GetFlag(FLAGS_num_games)) {
        break;
      }
    }
    if ((counter % 100) == 0) {
      LOG(INFO) << " Waiting for " + std::to_string(num_games) + " games in "
                << selfplay_model_dir << ". Found "
                << std::to_string(matched_output_files.size());
    }
    absl::SleepFor(absl::Milliseconds(50));
  }
  LOG(INFO) << "Done waiting. ";
}

void SampleTrainingExamples(std::string selfplay_dir, int window_size,
                            std::string golden_chunk_dir, int num_games,
                            double train_filter, int files_per_pattern) {
  // Set flags for sample_records.cc
  absl::SetFlag(&FLAGS_num_read_threads, 16);
  absl::SetFlag(&FLAGS_num_write_threads, 16);
  absl::SetFlag(&FLAGS_sample_frac, train_filter);
  absl::SetFlag(&FLAGS_seed, 0);
  absl::SetFlag(&FLAGS_shuffle, true);
  absl::SetFlag(&FLAGS_compression, 1);

  std::vector<std::string> selfplay_model_names =
      ListSelfplayDirs(selfplay_dir);
  if (window_size > selfplay_model_names.size()) {
    window_size = selfplay_model_names.size();
  }
  selfplay_model_names.resize(window_size);
  std::vector<std::string> src_patterns;
  src_patterns.reserve(window_size);
  for (const std::string& model_name : selfplay_model_names) {
    src_patterns.push_back(
        (fs::path(tensorflow::io::JoinPath(selfplay_dir, model_name))) /
        "*/*/*.tfrecord.zz");
  }
  int train_model_num = std::stoi(selfplay_model_names[0]) + 1;
  std::string train_model_name = absl::StrCat(
      std::string(6 - std::to_string(train_model_num).length(), '0'),
      train_model_num);

  std::vector<std::string> src_paths;
  int num_src_patterns = src_patterns.size();
  LOG(INFO) << "num_src_patterns is: " << num_src_patterns;

  for (int i = 0; i < num_src_patterns; ++i) {
    const auto& pattern = src_patterns[i];
    std::vector<std::string> paths;
    TF_CHECK_OK(tensorflow::Env::Default()->GetMatchingPaths(pattern, &paths));
    LOG(INFO) << pattern << " matched " << paths.size() << " files";
    if (files_per_pattern > 0) {
      MG_CHECK(static_cast<int>(paths.size()) >= files_per_pattern)
          << "require " << files_per_pattern << " files per pattern, "
          << pattern << " matched only " << paths.size();
      minigo::Random rnd(absl::GetFlag(FLAGS_seed),
                         minigo::Random::kUniqueStream);
      rnd.Shuffle(&paths);
      paths.resize(files_per_pattern);
    }
    for (auto& path : paths) {
      src_paths.push_back(std::move(path));
    }
  }
  std::string chunk_file_name = "{" + train_model_name + "}.tfrecord.zz";  //
  std::string dst_path =
      tensorflow::io::JoinPath(golden_chunk_dir, chunk_file_name);  //
  absl::SetFlag(&FLAGS_dst, dst_path);
  size_t num_records = minigo::Run(src_paths, FLAGS_dst);
  // Now, gather chunk_paths and num_examples info
  std::string chunk_pattern =
      fs::path(golden_chunk_dir) /
      absl::StrCat("{", train_model_name, "}-*-of-*.tfrecord.zz");
  std::vector<std::string> chunk_paths;
  TF_CHECK_OK(tensorflow::Env::Default()->GetMatchingPaths(chunk_pattern,
                                                           &chunk_paths));
  std::sort(chunk_paths.begin(), chunk_paths.end());
  QCHECK_EQ(chunk_paths.size(), absl::GetFlag(FLAGS_num_write_threads));
  chunk_paths.push_back(std::to_string(num_records));

  // Write to golden_chunk_list.txt
  std::string file_name =
      fs::path(golden_chunk_dir) /
      absl::StrCat(train_model_name, "-golden_chunk_list.txt");
  tensorflow::Status status;
  std::unique_ptr<tensorflow::WritableFile> file;
  status = tensorflow::Env::Default()->NewWritableFile(file_name, &file);
  if (!status.ok()) {
    LOG(ERROR) << "error opening " << file_name << " for write: " << status;
  }
  std::string concat_tfrecords;
  for (const auto& piece : chunk_paths) {
    concat_tfrecords += (piece + '\n');
  }
  status = file->Append({concat_tfrecords.data(), concat_tfrecords.size()});
  if (!status.ok()) {
    LOG(ERROR) << "error writing to " << file_name << ": " << status;
  }
  status = file->Close();
  if (!status.ok()) {
    LOG(ERROR) << "error closing " << file_name << ": " << status;
  }
}

}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  // List model dirs under selfplay dir in reverse order
  std::vector<std::string> model_names =
      minigo::ListSelfplayDirs(absl::GetFlag(FLAGS_selfplay_dir));
  if (model_names.empty()) {
    LOG(ERROR) << "Watcher couldn\'t find any selfplay games under "
               << absl::GetFlag(FLAGS_selfplay_dir) << "Either bootstrap.sh or "
               << "init_from_checkpoint.sh must be run before the train loop is"
               << "started";
  }
  int model_num = std::stoi(model_names[0]);
  while (model_num < absl::GetFlag(FLAGS_iterations)) {
    minigo::WaitForTrainingExamples(model_num, absl::GetFlag(FLAGS_num_games),
                                    absl::GetFlag(FLAGS_selfplay_dir));
    minigo::SampleTrainingExamples(
        absl::GetFlag(FLAGS_selfplay_dir), absl::GetFlag(FLAGS_window_size),
        absl::GetFlag(FLAGS_golden_chunk_dir), absl::GetFlag(FLAGS_num_games),
        absl::GetFlag(FLAGS_train_filter), absl::GetFlag(FLAGS_num_games));
    ++model_num;
  }
}
