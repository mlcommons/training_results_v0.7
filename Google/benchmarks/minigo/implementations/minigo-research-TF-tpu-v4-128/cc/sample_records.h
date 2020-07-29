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

#ifndef MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_SAMPLE_RECORDS_H_
#define MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_SAMPLE_RECORDS_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "base/commandlineflags.h"
#include "REDACTEDmemory/memory.h"
#include "REDACTEDstrings/match.h"
#include "REDACTEDstrings/str_format.h"
#include "REDACTEDstrings/string_view.h"
#include "REDACTEDstrings/strip.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/async/thread.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/init.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/logging.h"
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/random.h"
#include "third_party/tensorflow/core/lib/core/status.h"
#include "third_party/tensorflow/core/lib/io/record_reader.h"
#include "third_party/tensorflow/core/lib/io/record_writer.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/file_system.h"

DECLARE_double(sample_frac);
DECLARE_uint64(num_records);
DECLARE_int32(num_read_threads);
DECLARE_int32(num_write_threads);
DECLARE_int32(compression);
DECLARE_int32(files_per_pattern);
DECLARE_bool(shuffle);
DECLARE_string(dst);
DECLARE_uint64(seed);

namespace minigo {

class ReadThread : public Thread {
 public:
  struct Options {
    float sample_frac = 1;
  };

  ReadThread(std::vector<std::string> paths, const Options& options)
      : rnd_(FLAGS_seed, Random::kUniqueStream),
        paths_(std::move(paths)),
        options_(options) {}

  std::vector<tensorflow::tstring>& sampled_records() {
    return sampled_records_;
  }

  const std::vector<tensorflow::tstring>& sampled_records() const {
    return sampled_records_;
  }

 private:
  void Run() override;

  Random rnd_;
  const std::vector<std::string> paths_;
  std::vector<tensorflow::tstring> sampled_records_;
  const Options options_;
};

class WriteThread : public Thread {
 public:
  struct Options {
    int shard = 0;
    int num_shards = 1;
    int compression = 1;
  };

  WriteThread(std::vector<tensorflow::tstring> records, const std::string& path,
              const Options& options);

 private:
  void Run() override;

  std::string path_;
  std::vector<tensorflow::tstring> records_;
  const Options options_;
};

template <typename T>
void MoveAppend(std::vector<T>* src, std::vector<T>* dst) {
  if (dst->empty()) {
    *dst = std::move(*src);
  } else {
    std::move(std::begin(*src), std::end(*src), std::back_inserter(*dst));
    src->clear();
  }
}

std::vector<tensorflow::tstring> Read(std::vector<std::string> paths);

void Shuffle(std::vector<tensorflow::tstring>* records);

// void Write(std::vector<tensorflow::tstring> records, const std::string&
// path);
size_t Write(const std::vector<tensorflow::tstring>& records,
             const std::string& path);

// void Run(std::vector<std::string> src_paths, const std::string& dst_path);
size_t Run(const std::vector<std::string>& src_paths,
           const std::string& dst_path);
}  // namespace minigo

#endif  //  MLPERF_SUBMISSIONS_TRAINING_V0_7_MODELS_PROD_MINIGO_CC_SAMPLE_RECORDS_H_
