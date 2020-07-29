# Lint as: python3
"""Flume preprocessing pipeline for Criteo data.
"""

import logging as stdlogging
import os

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from REDACTED.pipeline.flume.py import runner

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_path", "",
    ("Input path. Be sure to set this to cover all data, to ensure "
     "that sparse vocabs are complete."))
flags.DEFINE_string(
    "output_path", "",
    "Output path.")
flags.DEFINE_string(
    "temp_dir", "",
    ("Directory to store temporary metadata. Important because vocab "
     "dictionaries will be stored here. Co-located with data, ideally."))
flags.DEFINE_string("csv_delimeter", "\t",
                    "Delimeter string for input and output.")
flags.DEFINE_bool(
    "vocab_gen_mode", False,
    ("If true, process full dataset and do not write CSV output. In this mode, "
     "See temp_dir for vocab files. input_path should cover all data, "
     "e.g. train, test, eval."))

NUMERIC_FEATURE_KEYS = ["int-feature-%d" % x for x in range(1, 14)]
CATEGORICAL_FEATURE_KEYS = ["categorical-feature-%d" % x for x in range(14, 40)]
LABEL_KEY = "clicked"

MAX_IND_RANGE = (40 * 1000 * 1000)

# Data is first preprocessed in pure Apache Beam using numpy.
# This removes missing values and hexadecimal-encoded values.
# For the TF schema, we can thus specify the schema as FixedLenFeature
# for TensorFlow Transform.
FEATURE_SPEC = dict([(name, tf.io.FixedLenFeature([], dtype=tf.int64))
                     for name in CATEGORICAL_FEATURE_KEYS] +
                    [(name, tf.io.FixedLenFeature([], dtype=tf.float32))
                     for name in NUMERIC_FEATURE_KEYS] +
                    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.float32))])
INPUT_METADATA = dataset_metadata.DatasetMetadata(
    dataset_schema.from_feature_spec(FEATURE_SPEC))


def apply_vocab_fn(inputs):
  """Preprocessing fn for sparse features.

  Applies vocab to bucketize sparse features. This function operates using
  previously-created vocab files.
  Pre-condition: Full vocab has been materialized.

  Args:
    inputs: Input features to transform.

  Returns:
    Output dict with transformed features.
  """
  outputs = {}

  outputs[LABEL_KEY] = inputs[LABEL_KEY]
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = inputs[key]
  for idx, key in enumerate(CATEGORICAL_FEATURE_KEYS):
    vocab_fn = os.path.join(
        FLAGS.temp_dir, "tftransform_tmp", "feature_{}_vocab".format(idx))
    outputs[key] = tft.apply_vocabulary(inputs[key], vocab_fn)

  return outputs


def compute_vocab_fn(inputs):
  """Preprocessing fn for sparse features.

  This function computes unique IDs for the sparse features. We rely on implicit
  behavior which writes the vocab files to the vocab_filename specified in
  tft.compute_and_apply_vocabulary.

  Pre-condition: Sparse features have been converted to integer and mod'ed with
  MAX_IND_RANGE.

  Args:
    inputs: Input features to transform.

  Returns:
    Output dict with transformed features.
  """
  outputs = {}

  outputs[LABEL_KEY] = inputs[LABEL_KEY]
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = inputs[key]
  for idx, key in enumerate(CATEGORICAL_FEATURE_KEYS):
    outputs[key] = tft.compute_and_apply_vocabulary(
        x=inputs[key],
        vocab_filename="feature_{}_vocab".format(idx))

  return outputs


class FillMissing(beam.DoFn):
  """Fills missing elements with zero string value."""

  def process(self, element):
    elem_list = element.split(FLAGS.csv_delimeter)
    out_list = []
    for val in elem_list:
      new_val = "0" if not val else val
      out_list.append(new_val)
    yield (FLAGS.csv_delimeter).join(out_list)


class NegsToZeroLog(beam.DoFn):
  """For int features, sets negative values to zero and takes log(x+1)."""

  def process(self, element):
    elem_list = element.split(FLAGS.csv_delimeter)
    out_list = []
    for i, val in enumerate(elem_list):
      if i > 0 and i < 14:
        new_val = "0" if int(val) < 0 else val
        new_val = np.log(int(new_val) + 1)
        new_val = str(new_val)
      else:
        new_val = val
      out_list.append(new_val)
    yield (FLAGS.csv_delimeter).join(out_list)


class HexToIntModRange(beam.DoFn):
  """For categorical features, takes decimal value and mods with max value."""

  def process(self, element):
    elem_list = element.split(FLAGS.csv_delimeter)
    out_list = []
    for i, val in enumerate(elem_list):
      if i > 13:
        # new_val = int(val, 16)
        new_val = int(val, 16) % MAX_IND_RANGE
      else:
        new_val = val
      out_list.append(str(new_val))
    yield (FLAGS.csv_delimeter).join(out_list)


def process_files(data_path, output_path):
  """Returns a pipeline which preprocesses Criteo data.

  Two processing modes are supported. Raw data will require two passes.
  If full vocab files already exist, only one pass is necessary.

  Args:
    data_path: File(s) to read.
    output_path: Path to which output CSVs are written, if necessary.
  """

  def pipeline(root):
    """Pipeline instantiation function.

    Args:
      root: Source pipeline from which to extend.
    """

    preprocessing_fn = compute_vocab_fn if FLAGS.vocab_gen_mode else apply_vocab_fn

    with tft_beam.Context(temp_dir=FLAGS.temp_dir):
      processed_lines = (
          root
          # Read in TSV data.
          | beam.io.ReadFromText(data_path)
          # Fill in missing elements with the defaults (zeros).
          | "FillMissing" >> beam.ParDo(FillMissing())
          # For numerical features, set negatives to zero. Then take log(x+1).
          | "NegsToZeroLog" >> beam.ParDo(NegsToZeroLog())
          # For categorical features, mod the values with vocab size.
          | "HexToIntModRange" >> beam.ParDo(HexToIntModRange()))

      # CSV reader: List the cols in order, as dataset schema is not ordered.
      ordered_columns = [LABEL_KEY
                        ] + NUMERIC_FEATURE_KEYS + CATEGORICAL_FEATURE_KEYS
      converter = tft.coders.CsvCoder(
          ordered_columns, INPUT_METADATA.schema, delimiter=FLAGS.csv_delimeter)

      converted_data = (
          processed_lines
          | "DecodeData" >> beam.Map(converter.decode))

      transformed_dataset, transform_fn = (  # pylint: disable=unused-variable
          (converted_data, INPUT_METADATA)
          | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset

      if not FLAGS.vocab_gen_mode:
        # Write to CSV.
        transformed_csv_coder = tft.coders.CsvCoder(
            ordered_columns, transformed_metadata.schema,
            delimiter=FLAGS.csv_delimeter)
        _ = (
            transformed_data
            | "EncodeDataCsv" >> beam.Map(transformed_csv_coder.encode)
            | "WriteDataCsv" >> beam.io.WriteToText(output_path))

  return pipeline


def main(argv):
  del argv
  stdlogging.getLogger().setLevel(stdlogging.INFO)
  runner.program_started()  # Must be called before creating the pipeline.

  pipeline = process_files(FLAGS.input_path, FLAGS.output_path)
  runner.FlumeRunner().run(pipeline).wait_until_finish()


if __name__ == "__main__":
  app.run(main)
