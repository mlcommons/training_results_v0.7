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

#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/directory_watcher.h"

#include <utility>
#include <vector>

#include "base/logging.h"
#include "REDACTEDmemory/memory.h"
#include "REDACTEDstrings/str_cat.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/poll_thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/path.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/file/utils.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "third_party/tensorflow/core/lib/io/path.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/file_system.h"

namespace minigo {

namespace {

bool ParseModelPathPattern(const std::string& pattern, std::string* directory,
                           std::string* basename_pattern) {
  if (pattern.find("%d/saved_model.pb") != std::string::npos) {
    *directory = std::string(
        file::SplitPath(std::string(file::SplitPath(pattern).first)).first);
    *basename_pattern = "*/saved_model.pb";
  } else {
    auto pair = file::SplitPath(pattern);

    *directory = std::string(pair.first);
    if (directory->find('%') != std::string::npos ||
        directory->find('*') != std::string::npos) {
      MG_LOG(ERROR) << "invalid pattern \"" << pattern
                    << "\": directory part must not contain '*' or '%'";
      return false;
    }
    if (directory->empty()) {
      MG_LOG(ERROR) << "directory not be empty";
      return false;
    }

    *basename_pattern = std::string(pair.second);
    auto it = basename_pattern->find('%');
    if (it == std::string::npos || basename_pattern->find("%d") != it ||
        basename_pattern->rfind("%d") != it) {
      MG_LOG(ERROR) << "invalid pattern \"" << pattern
                    << "\": basename must contain "
                    << " exactly one \"%d\" and no other matchers";
      return false;
    }
  }
  return true;
}

bool MatchBasename(const std::string& basename, const std::string& pattern,
                   int* generation) {
  int gen = 0;
  int n = 0;
  if (sscanf(basename.c_str(), pattern.c_str(), &gen, &n) != 1 ||
      n != static_cast<int>(basename.size())) {
    return false;
  }
  *generation = gen;
  return true;
}

}  // namespace

DirectoryWatcher::DirectoryWatcher(
    const std::string& pattern, absl::Duration poll_interval,
    std::function<void(const std::string&)> callback)
    : callback_(std::move(callback)) {
  MG_CHECK(ParseModelPathPattern(pattern, &directory_, &basename_pattern_));
  basename_and_length_pattern_ = absl::StrCat(basename_pattern_, "%n");

  poll_thread_ = absl::make_unique<PollThread>(
      "DirWatcher", poll_interval, std::bind(&DirectoryWatcher::Poll, this));
  poll_thread_->Start();
}

DirectoryWatcher::~DirectoryWatcher() {
  poll_thread_->Join();
}

void DirectoryWatcher::Poll() {
  const std::string* latest_basename = nullptr;
  std::string new_latest_path;
  if (basename_pattern_.find("saved_model.pb") != std::string::npos) {
    std::vector<std::string> matched_entries;
    TF_CHECK_OK(tensorflow::Env::Default()->GetMatchingPaths(
        file::JoinPath(directory_, basename_pattern_), &matched_entries));
    // Find the model directory with the largest integer name.
    std::sort(matched_entries.begin(), matched_entries.end(),
              std::greater<std::string>());
    if (matched_entries.empty()) {
      return;
    }
    new_latest_path = std::string(file::SplitPath(matched_entries[0]).first);
  } else {
    // List all the files in the given directory.
    std::vector<std::string> basenames;
    if (!file::ListDir(directory_, &basenames)) {
      return;
    }
    // Find the file basename that contains the largest integer.
    int latest_generation = -1;
    for (const auto& basename : basenames) {
      int generation = 0;
      if (!MatchBasename(basename, basename_and_length_pattern_, &generation)) {
        continue;
      }
      if (latest_basename == nullptr || generation > latest_generation) {
        latest_basename = &basename;
        latest_generation = generation;
      }
    }

    if (latest_basename == nullptr) {
      // Didn't find any matching files.
      return;
    }

    // Build the full path to the latest model.
    new_latest_path = file::JoinPath(directory_, *latest_basename);
  }
  if (new_latest_path == latest_path_) {
    // The latest path hasn't changed.
    return;
  }

  // Update the latest known path and invoke the callback.
  latest_path_ = std::move(new_latest_path);
  callback_(latest_path_);
}

}  // namespace minigo
