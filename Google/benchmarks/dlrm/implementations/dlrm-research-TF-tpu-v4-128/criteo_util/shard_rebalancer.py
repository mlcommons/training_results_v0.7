# Lint as: python3
"""Rebalance a set of CSV/TFRecord shards to a target number of files.
"""

import logging as stdlogging

from absl import app
from absl import flags
import apache_beam as beam
import tensorflow.compat.v1 as tf

from REDACTED.pipeline.flume.py import runner

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_path", "",
    ("Input path. Be sure to set this to cover all data, to ensure "
     "that sparse vocabs are complete."))
flags.DEFINE_string(
    "output_path", "",
    "Output path.")
flags.DEFINE_integer(
    "num_output_files", -1,
    "Number of output file shards.")
flags.DEFINE_string(
    "filetype", "tfrecord",
    "One of {tfrecord, csv}.")

NUMERIC_FEATURE_KEYS = ["int-feature-%d" % x for x in range(1, 14)]
CATEGORICAL_FEATURE_KEYS = ["categorical-feature-%d" % x for x in range(14, 40)]
LABEL_KEY = "clicked"


# Data is first preprocessed in pure Apache Beam using numpy.
# This removes missing values and hexadecimal-encoded values.
# For the TF schema, we can thus specify the schema as FixedLenFeature
# for TensorFlow Transform.
FEATURE_SPEC = dict([(name, tf.io.FixedLenFeature([], dtype=tf.int64))
                     for name in CATEGORICAL_FEATURE_KEYS] +
                    [(name, tf.io.FixedLenFeature([], dtype=tf.float32))
                     for name in NUMERIC_FEATURE_KEYS] +
                    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.float32))])


def process_files(data_path, output_path):
  """Returns a pipeline which rebalances data shards.

  Args:
    data_path: File(s) to read.
    output_path: Path to which output CSVs are written, if necessary.
  """

  def csv_pipeline(root):

    _ = (
        root
        | beam.io.ReadFromText(data_path)
        | beam.io.WriteToText(output_path,
                              num_shards=FLAGS.num_output_files))

  def tfrecord_pipeline(root):
    """Pipeline instantiation function.

    Args:
      root: Source pipeline from which to extend.
    """

    example_coder = beam.coders.ProtoCoder(tf.train.Example)
    _ = (
        root
        | beam.io.ReadFromTFRecord(data_path, coder=example_coder)
        | beam.io.WriteToTFRecord(output_path, file_name_suffix="tfrecord",
                                  coder=example_coder,
                                  num_shards=FLAGS.num_output_files))

  pipeline = tfrecord_pipeline if FLAGS.filetype == "tfrecord" else csv_pipeline

  return pipeline


def main(argv):
  del argv
  if FLAGS.num_output_files < 1:
    raise ValueError("Number of output shards must be defined.")
  stdlogging.getLogger().setLevel(stdlogging.INFO)
  runner.program_started()  # Must be called before creating the pipeline.

  pipeline = process_files(FLAGS.input_path, FLAGS.output_path)
  runner.FlumeRunner().run(pipeline).wait_until_finish()


if __name__ == "__main__":
  app.run(main)
