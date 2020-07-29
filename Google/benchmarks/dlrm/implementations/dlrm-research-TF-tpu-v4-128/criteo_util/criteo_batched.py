# Lint as: python3
"""Flume preprocessing pipeline for Criteo data.
"""

import collections
import csv
import logging as stdlogging
import re

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.pipeline.flume.py import runner


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_path", "",
    ("Input path. Be sure to set this to cover all data, to ensure "
     "that sparse vocabs are complete."))
flags.DEFINE_string(
    "output_path", "",
    "Output directory and prefix.")
flags.DEFINE_integer(
    "batch_size", 4,
    "Number of samples to group into a batch.")
flags.DEFINE_bool(
    "drop_remainder", False,
    ("If true, drop remainder elements that cannot compose a full batch. "
     "If false, pad the remainder."))

NUMERIC_FEATURE_KEYS = ["int-feature-%d" % x for x in range(1, 14)]
CATEGORICAL_FEATURE_KEYS = ["categorical-feature-%d" % x for x in range(14, 40)]
LABEL_KEY = "clicked"
FIELDS = [LABEL_KEY] + NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS

num_samples = 0
num_files = 0


def gen_batches(input_filename):
  """Function to be used by beam.Map()."""

  print("Worker processing file {}".format(input_filename))

  def _output_file_index(s):
    nums = re.findall(r"\d+", s)
    if len(nums) == 1:
      return nums[0]
    elif len(nums) > 1:
      return nums[-2]+"-of-"+nums[-1]
    else:
      idx = abs(hash(s)) % (10 ** 8)
      return str(idx)

  out_filename = FLAGS.output_path + _output_file_index(input_filename)

  def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

  # def _int64_feature(value):
  #   return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  def _serialize_batch(feature_dict):
    for key in feature_dict:
      if key in CATEGORICAL_FEATURE_KEYS:
        # feature_dict[key] = _int64_feature(feature_dict[key])

        # Serialize-to-string for int32s.
        array_int32 = np.array(feature_dict[key], dtype=np.int32)
        feature_dict[key] = _bytes_feature(
            [memoryview(array_int32).tobytes()])
      else:
        feature_dict[key] = _float_feature(feature_dict[key])
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()

  def _extract_batch(dict_reader, batch_size):
    """dict_reader is a csv.DictReader."""
    batch_features = collections.defaultdict(list)
    idx = 0
    for idx, sample in enumerate(dict_reader):
      for key in sample:
        if key in CATEGORICAL_FEATURE_KEYS:
          batch_features[key].append(int(sample[key]))
        else:
          batch_features[key].append(float(sample[key]))
      if (idx + 1) % batch_size == 0:
        yield batch_size, batch_features
        batch_features.clear()
    # If less than a batch remains, return the number of valid entries in batch.
    yield (idx+1) % batch_size, batch_features

  def _pad_to_batch(feature_dict, samples_written):
    pad_size = FLAGS.batch_size - samples_written
    for key in feature_dict:
      if key in CATEGORICAL_FEATURE_KEYS:
        feature_dict[key].extend([0] * pad_size)
      elif key in NUMERIC_FEATURE_KEYS:
        feature_dict[key].extend([0.0] * pad_size)
      elif key == LABEL_KEY:
        feature_dict[key].extend([-1.0] * pad_size)
      else:
        raise ValueError("Unknown feature key while padding to batch.")
    return feature_dict

  num_batches = 0
  with tf.io.gfile.GFile(input_filename, "r") as f:
    dict_reader = csv.DictReader(f, dialect=csv.excel_tab, fieldnames=FIELDS)
    with tf.io.TFRecordWriter(out_filename) as writer:
      for samples_written, features_batch in _extract_batch(
          dict_reader, FLAGS.batch_size):
        if samples_written < FLAGS.batch_size:
          if FLAGS.drop_remainder:
            continue
          else:
            features_batch = _pad_to_batch(features_batch, samples_written)
        example_bytes = _serialize_batch(features_batch)
        writer.write(example_bytes)
        num_batches += 1

  return num_batches


def process_files(input_path):
  """Returns a pipeline which creates batched TFRecords.

  Args:
    input_path: File pattern to read.
  """

  def pipeline(root):
    """Pipeline instantiation function.

    Args:
      root: Source pipeline from which to extend.
    """
    global num_files

    filename_list = tf.io.gfile.glob(input_path)
    num_files = len(filename_list)
    assert num_files > 0, "No files provided."
    print("Beginning processing on {} files.".format(num_files))

    def capture_sample_cnt(line):
      global num_samples
      num_samples = int(line)
      print("\n** Number of samples: {}".format(num_samples))

    _ = (
        root
        | beam.io.ReadFromText(input_path)
        | beam.combiners.Count.Globally()
        | beam.Map(capture_sample_cnt))

    def _print_output(num_batches):
      num_batches = int(num_batches)
      print("\n** {} batches created ({} samples).\n".format(
          num_batches, num_batches * FLAGS.batch_size))

    _ = (
        root
        | "CreateFilenameList" >> beam.Create(filename_list)
        | "GenerateBatches" >> beam.Map(gen_batches)
        | "AccumulateNumBatches" >> beam.CombineGlobally(sum)
        | "PrintReport" >> beam.Map(_print_output))

  return pipeline


def main(argv):
  del argv
  stdlogging.getLogger().setLevel(stdlogging.INFO)
  runner.program_started()  # Must be called before creating the pipeline.

  pipeline = process_files(FLAGS.input_path)
  runner.FlumeRunner().run(pipeline).wait_until_finish()


if __name__ == "__main__":
  app.run(main)
