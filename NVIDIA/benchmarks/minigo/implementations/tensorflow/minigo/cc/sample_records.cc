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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <list>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "cc/async/thread.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/random.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace minigo {

class ReadThread : public Thread {
 public:
  struct Options {
    float sample_frac = 1;
  };

  ReadThread(uint64_t seed, std::vector<std::string> paths, const Options& options)
      : rnd_(seed, Random::kUniqueStream),
        paths_(std::move(paths)),
        options_(options) {}

  std::vector<std::string>& sampled_records() { return sampled_records_; }
  const std::vector<std::string>& sampled_records() const {
    return sampled_records_;
  }

 private:
  void Run() override {
    tensorflow::io::RecordReaderOptions options;
    for (const auto& path : paths_) {
      std::unique_ptr<tensorflow::RandomAccessFile> file;
      TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(path, &file));
      if (absl::EndsWith(path, ".zz")) {
        options.compression_type =
            tensorflow::io::RecordReaderOptions::ZLIB_COMPRESSION;
      } else {
        options.compression_type = tensorflow::io::RecordReaderOptions::NONE;
      }

      tensorflow::io::SequentialRecordReader reader(file.get(), options);
      std::string record;
      for (;;) {
        auto status = reader.ReadRecord(&record);
        if (status.code() == tensorflow::error::OUT_OF_RANGE) {
          // Reached the end of the file.
          break;
        } else if (!status.ok()) {
          // Some other error.
          MG_LOG(WARNING) << "Error reading record from \"" << path
                          << "\": " << status;
          continue;
        }

        if (options_.sample_frac == 1 || rnd_() < options_.sample_frac) {
          sampled_records_.push_back(std::move(record));
        }
      }
    }
  }

  Random rnd_;
  const std::vector<std::string> paths_;
  std::vector<std::string> sampled_records_;
  const Options options_;
};

class WriteThread : public Thread {
 public:
  struct Options {
    int shard = 0;
    int num_shards = 1;
    int compression = 1;
  };

  WriteThread(std::vector<std::string> records, std::string path,
              const Options& options)
      : records_(std::move(records)), options_(options) {
    if (options_.num_shards == 1) {
      path_ = path;
    } else {
      absl::string_view expected_ext =
          options_.compression == 0 ? ".tfrecord" : ".tfrecord.zz";
      absl::string_view stem = path;
      MG_CHECK(absl::ConsumeSuffix(&stem, expected_ext))
          << "expected path to have extension '" << expected_ext
          << "', got '" << stem << "'";
      path_ = absl::StrFormat("%s-%05d-of-%05d.tfrecord.zz", stem,
                              options_.shard, options_.num_shards);
    }
  }

 private:
  void Run() override {
    std::unique_ptr<tensorflow::WritableFile> file;
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(path_, &file));

    tensorflow::io::RecordWriterOptions options;
    if (options_.compression > 0) {
      MG_CHECK(options_.compression <= 9);
      options.compression_type =
          tensorflow::io::RecordWriterOptions::ZLIB_COMPRESSION;
      options.zlib_options.compression_level = options_.compression;
    } else {
      options.compression_type = tensorflow::io::RecordWriterOptions::NONE;
    }

    tensorflow::io::RecordWriter writer(file.get(), options);
    for (const auto& record : records_) {
      TF_CHECK_OK(writer.WriteRecord(record));
    }

    TF_CHECK_OK(writer.Close());
    TF_CHECK_OK(file->Close());
  }

  std::string path_;
  std::vector<std::string> records_;
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

std::vector<std::string> Read(std::vector<std::string> paths, 
                              uint64_t seed,
                              int32_t n_read_threads,
                              double sample_fraction,
                              int32_t& n_sampled_records,
                              bool verbose) {
  int num_paths = static_cast<int>(paths.size());
  int num_read_threads = std::min<int>(n_read_threads, num_paths);

  if(verbose) {
    MG_LOG(INFO) << "reading " << num_paths << " files on " << num_read_threads
                 << " threads";
  }

  ReadThread::Options read_options;
  // If --sample_frac wasn't set, default to reading all records: we need to
  // read all records from all files in order to fairly read exaclty
  // --num_records records.
  read_options.sample_frac = sample_fraction == 0 ? 1 : sample_fraction;

  std::vector<std::unique_ptr<ReadThread>> threads;
  for (int i = 0; i < num_read_threads; ++i) {
    // Get the record paths that this thread should run on.
    int begin = i * num_paths / num_read_threads;
    int end = (i + 1) * num_paths / num_read_threads;
    std::vector<std::string> thread_paths;
    for (int j = begin; j < end; ++j) {
      thread_paths.push_back(std::move(paths[j]));
    }
    threads.push_back(
        absl::make_unique<ReadThread>(seed, std::move(thread_paths), read_options));
  }
  for (auto& t : threads) {
    t->Start();
  }
  for (auto& t : threads) {
    t->Join();
  }

  // Concatenate sampled records.
  size_t n = 0;
  for (const auto& t : threads) {
    n += t->sampled_records().size();
  }

  n_sampled_records = n;
  if(verbose) {
    MG_LOG(INFO) << "sampled " << n << " records" << " r2= " << n_sampled_records;
    MG_LOG(INFO) << "concatenating";
  }
  std::vector<std::string> records;
  records.reserve(n);
  for (const auto& t : threads) {
    MoveAppend(&t->sampled_records(), &records);
  }

  return records;
}

void Shuffle(uint64_t seed, std::vector<std::string>* records, bool verbose) {
  Random rnd(seed, Random::kUniqueStream);
  if(verbose) MG_LOG(INFO) << "shuffling";
  rnd.Shuffle(records);
}

void Write(std::vector<std::string> records, const std::string& path,
           int32_t n_write_threads, int32_t compression,
           int32_t n_records,
           bool verbose) {
  WriteThread::Options write_options;
  write_options.num_shards = n_write_threads;
  write_options.compression = compression;

  size_t num_records;
  if (n_records != 0) {
    // TODO(tommadams): add support for either duplicating some records or allow
    // fewer than requested number of records to be written.
    MG_CHECK(n_records <= records.size())
        << "--num_records=" << n_records << " but there are only "
        << records.size() << " available";
    num_records = n_records;
  } else {
    num_records = static_cast<size_t>(records.size());
  }

  size_t total_dst = 0;
  std::vector<std::unique_ptr<WriteThread>> threads;
  for (int shard = 0; shard < n_write_threads; ++shard) {
    write_options.shard = shard;

    // Calculate the range of source records for this shard.
    size_t begin_src = shard * records.size() / n_write_threads;
    size_t end_src = (shard + 1) * records.size() / n_write_threads;
    size_t num_src = end_src - begin_src;

    // Calculate the number of destination records for this shard.
    size_t begin_dst = shard * num_records / n_write_threads;
    size_t end_dst = (shard + 1) * num_records / n_write_threads;
    size_t num_dst = end_dst - begin_dst;

    total_dst += num_dst;

    // Sample the records for this shard.
    std::vector<std::string> shard_records;
    shard_records.reserve(num_dst);
    for (size_t i = 0; i < num_dst; ++i) {
      size_t j = begin_src + i * num_src / num_dst;
      shard_records.push_back(std::move(records[j]));
    }

    threads.push_back(absl::make_unique<WriteThread>(std::move(shard_records),
                                                     path, write_options));
  }

  MG_CHECK(total_dst == num_records);
  if(verbose) {
    MG_LOG(INFO) << "writing " << num_records << " records to " << path;
  }
  for (auto& t : threads) {
    t->Start();
  }
  for (auto& t : threads) {
    t->Join();
  }
}

int32_t Run(double sample_fraction,
            uint64_t num_records,
            int32_t num_read_threads,
            int32_t num_write_threads,
            int32_t compression,
            bool shuffle,
            uint64_t seed,
            std::vector<std::string> src_paths, const std::string& dst_path,
            bool verbose) {

  MG_CHECK((sample_fraction != 0) != (num_records != 0))
      << "expected exactly one of --sample_frac and --num_records to be "
         "non-zero";

  MG_CHECK(!src_paths.empty());
  MG_CHECK(!dst_path.empty());

  int32_t n_sampled_records = 0;
  auto records = Read(std::move(src_paths), seed, num_read_threads, sample_fraction, n_sampled_records, verbose);

  if (shuffle) {
    Shuffle(seed, &records, verbose);
  }

  Write(std::move(records), dst_path, num_write_threads, compression, num_records, verbose);

  return n_sampled_records;
}

namespace sample_records {

int32_t run(double sample_fraction,
            uint64_t num_records,
            int32_t num_read_threads,
            int32_t num_write_threads,
            int32_t compression,
            int32_t files_per_pattern,
            bool shuffle,
            uint64_t seed,
            std::list<std::string> src_patterns,
            std::string dst,
            bool verbose) {

  std::vector<std::string> src_paths;
  for (const auto& pattern: src_patterns) {
    std::vector<std::string> paths;
    TF_CHECK_OK(tensorflow::Env::Default()->GetMatchingPaths(pattern, &paths));
    if(verbose) MG_LOG(INFO) << pattern << " matched " << paths.size() << " files";
    if (files_per_pattern > 0) {
      MG_CHECK(static_cast<int>(paths.size()) >= files_per_pattern)
          << "require " << files_per_pattern << " files per pattern, "
          << pattern << " matched only " << paths.size();
      minigo::Random rnd(seed, minigo::Random::kUniqueStream);
      rnd.Shuffle(&paths);
      paths.resize(files_per_pattern);
    }
    for (auto& path : paths) {
      src_paths.push_back(std::move(path));
    }
  }

  return minigo::Run (sample_fraction,
                      num_records,
                      num_read_threads,
                      num_write_threads,
                      compression,
                      shuffle,
                      seed,
                      std::move(src_paths), dst,
                      verbose);
}


} // namespace sample_records 
} // namespace minigo
