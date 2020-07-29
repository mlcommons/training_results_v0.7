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
#include "REDACTEDsubmissions/training/v0_7/models/prod/minigo/cc/sample_records.h"

#include <cstddef>

DEFINE_double(sample_frac, 0,
              "Fraction of records to read. Exactly one of sample_frac or "
              "num_records must be non-zero.");
DEFINE_uint64(num_records, 0,
              "Exact number of records to sample. Exactly one of sample_frac "
              "or num_records must be non-zero.");
DEFINE_int32(num_read_threads, 1,
             "Number of threads to use when reading source files.");
DEFINE_int32(num_write_threads, 1,
             "Number of threads to use when writing destination files. If "
             "num_write threads is > 1, the destination file will be sharded "
             "with one shard per write thread. Shards will be named "
             "<basename>-NNNNN-of-NNNNN.tfrecord.zz");
DEFINE_int32(compression, 1,
             "Compression level between 0 (disabled) and 9. Default is 1.");
DEFINE_int32(files_per_pattern, 0,
             "If > 0, randomly select exactly files_per_pattern input files "
             "to sample records from, failing if fewer are found.");
DEFINE_uint64(seed, 0, "Random seed.");
DEFINE_bool(shuffle, false, "Whether to shuffle the sampled records.");
DEFINE_string(dst, "",
              "Destination path. If path has a .zz suffix, the file will be "
              "automatically compressed.");
namespace minigo {

void ReadThread::Run() {
  namespace tf_io = ::tensorflow::io;
  tf_io::RecordReaderOptions options;
  for (const auto& path : paths_) {
    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(path, &file));
    if (absl::EndsWith(path, ".zz")) {
      options.compression_type = tf_io::RecordReaderOptions::ZLIB_COMPRESSION;
    } else {
      options.compression_type = tf_io::RecordReaderOptions::NONE;
    }

    tensorflow::io::SequentialRecordReader reader(file.get(), options);
    tensorflow::tstring record;
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

WriteThread::WriteThread(std::vector<tensorflow::tstring> records,
                         const std::string& path, const Options& options)
    : records_(std::move(records)), options_(options) {
  if (options_.num_shards == 1) {
    path_ = path;
  } else {
    absl::string_view expected_ext =
        options_.compression == 0 ? ".tfrecord" : ".tfrecord.zz";
    absl::string_view stem = path;
    MG_CHECK(absl::ConsumeSuffix(&stem, expected_ext))
        << "expected path to have extension '" << expected_ext << "', got '"
        << stem << "'";
    path_ = absl::StrFormat("%s-%05d-of-%05d.tfrecord.zz", stem, options_.shard,
                            options_.num_shards);
  }
}

void WriteThread::Run() {
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

std::vector<tensorflow::tstring> Read(std::vector<std::string> paths) {
  int num_paths = static_cast<int>(paths.size());
  int num_read_threads = std::min<int>(FLAGS_num_read_threads, num_paths);

  MG_LOG(INFO) << "reading " << num_paths << " files on " << num_read_threads
               << " threads";

  ReadThread::Options read_options;
  // If --sample_frac wasn't set, default to reading all records: we need to
  // read all records from all files in order to fairly read exaclty
  // --num_records records.
  read_options.sample_frac = FLAGS_sample_frac == 0 ? 1 : FLAGS_sample_frac;

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
        absl::make_unique<ReadThread>(std::move(thread_paths), read_options));
  }
  for (auto& t : threads) {
    t->Start();
  }
  for (auto& t : threads) {
    t->Join();
  }

  // Concatenate sampled records.
  size_t num_sampled = 0;
  for (const auto& t : threads) {
    num_sampled += t->sampled_records().size();
  }
  MG_LOG(INFO) << "sampled " << num_sampled << " records";
  MG_LOG(INFO) << "concatenating";
  std::vector<tensorflow::tstring> records;
  records.reserve(num_sampled);
  for (const auto& t : threads) {
    MoveAppend(&t->sampled_records(), &records);
  }

  return records;
}

void Shuffle(std::vector<tensorflow::tstring>* records) {
  Random rnd(FLAGS_seed, Random::kUniqueStream);
  MG_LOG(INFO) << "shuffling";
  rnd.Shuffle(records);
}

size_t Write(const std::vector<tensorflow::tstring>& records,
             const std::string& path) {
  MG_LOG(INFO) << "writing to " << path;

  WriteThread::Options write_options;
  write_options.num_shards = FLAGS_num_write_threads;
  write_options.compression = FLAGS_compression;

  size_t num_records;
  if (FLAGS_num_records != 0) {
    MG_CHECK(FLAGS_num_records <= records.size())
        << "--num_records=" << FLAGS_num_records << " but there are only "
        << records.size() << " available";
    num_records = FLAGS_num_records;
  } else {
    num_records = static_cast<size_t>(records.size());
  }

  size_t total_dst = 0;
  std::vector<std::unique_ptr<WriteThread>> threads;
  for (int shard = 0; shard < FLAGS_num_write_threads; ++shard) {
    write_options.shard = shard;

    // Calculate the range of source records for this shard.
    size_t begin_src = shard * records.size() / FLAGS_num_write_threads;
    size_t end_src = (shard + 1) * records.size() / FLAGS_num_write_threads;
    size_t num_src = end_src - begin_src;

    // Calculate the number of destination records for this shard.
    size_t begin_dst = shard * num_records / FLAGS_num_write_threads;
    size_t end_dst = (shard + 1) * num_records / FLAGS_num_write_threads;
    size_t num_dst = end_dst - begin_dst;

    total_dst += num_dst;

    // Sample the records for this shard.
    std::vector<tensorflow::tstring> shard_records;
    shard_records.reserve(num_dst);
    for (size_t i = 0; i < num_dst; ++i) {
      size_t j = begin_src + i * num_src / num_dst;
      shard_records.push_back(std::move(records[j]));
    }

    threads.push_back(absl::make_unique<WriteThread>(std::move(shard_records),
                                                     path, write_options));
  }

  MG_CHECK(total_dst == num_records);
  MG_LOG(INFO) << "writing " << num_records << " records to " << path;
  for (auto& t : threads) {
    t->Start();
  }
  for (auto& t : threads) {
    t->Join();
  }
  return num_records;
}

size_t Run(const std::vector<std::string>& src_paths,
           const std::string& dst_path) {
  MG_CHECK((FLAGS_sample_frac != 0) != (FLAGS_num_records != 0))
      << "expected exactly one of --sample_frac and --num_records to be "
         "non-zero";

  MG_CHECK(!src_paths.empty());
  MG_CHECK(!dst_path.empty());

  auto records = Read(std::move(src_paths));

  if (FLAGS_shuffle) {
    Shuffle(&records);
  }

  size_t num_records = Write(records, dst_path);

  MG_LOG(INFO) << "Done writing";

  return num_records;
}

}  // namespace minigo
