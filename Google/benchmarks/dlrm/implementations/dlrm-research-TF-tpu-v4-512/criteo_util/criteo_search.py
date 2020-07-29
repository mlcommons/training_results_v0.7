# Lint as: python3
"""Search all data for features which meet a certain criteria."""

import logging as stdlogging

from absl import app
from absl import flags
import apache_beam as beam
import tensorflow_transform.beam as tft_beam

import runner

FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", "//jn-d/home/tayo/tmp/auc.py", (
    "Input path to be searched. Be sure to set this to cover all data, to ensure "
    "that sparse vocabs are complete."))
flags.DEFINE_string("output_path", "/tmp/counts", "Output path.")
flags.DEFINE_string(
    "temp_dir",
    "//mb-d/home/tpu-perf-team/tayo/criteo/criteo_preprocessed_tmp/",
    "Directory to store temporary metadata. Co-located with data, ideally.")
flags.DEFINE_string("csv_delimeter", "\t",
                    "Delimeter string for input and output.")

MAX_IND_RANGE = (40 * 1000 * 1000)


class HexSearchFilter(beam.DoFn):
  """For categorical features, emit samples with categorical features found in a list of given demical numbers when mod'ed with MAX_IND_RANGE.
  """

  def process(self, element, feature_idx, decimal_list):
    search_set = set(decimal_list)
    elem_list = element.split(FLAGS.csv_delimeter)

    if len(elem_list) != 40:
      yield beam.pvalue.TaggedOutput("malformed_entries", element)

    raw_elem = elem_list[14 + feature_idx]
    if int(raw_elem, 16) % MAX_IND_RANGE in search_set:
      yield element


def process_files(data_path, output_path):
  """Returns a pipeline which searches Criteo data.

  Searches Criteo samples for particular values or malformed entries.

  Args:
    data_path: File(s) to read.
    output_path: Path to which output CSVs are written, if necessary.
  """

  def pipeline(root):
    """Pipeline instantiation function.

    Args:
      root: Source pipeline from which to extend.
    """

    # This pipeline is concerned only with searching the sparse features.

    with tft_beam.Context(temp_dir=FLAGS.temp_dir):
      processed_lines = (
          root
          # Read in TSV data.
          | "ReadData" >> beam.io.ReadFromText(data_path)
          # For categorical features, search for the given values, as integers.
          | "HexSearchFilter" >> beam.ParDo(HexSearchFilter(), 1, [
              14198776, 26023586, 21084594
          ]).with_outputs("malformed_entries", main="filtered_outputs"))

      malformed_lines = processed_lines.malformed_entries
      processed_lines = processed_lines.filtered_outputs

      _ = (processed_lines | "WriteData" >> beam.io.WriteToText(output_path))

      _ = (
          malformed_lines
          | "WriteDataMalformed" >>
          beam.io.WriteToText(output_path + "_malformed"))

  return pipeline


def main(argv):
  del argv
  stdlogging.getLogger().setLevel(stdlogging.INFO)
  runner.program_started()  # Must be called before creating the pipeline.

  pipeline = process_files(FLAGS.input_path, FLAGS.output_path)
  runner.FlumeRunner().run(pipeline).wait_until_finish()


if __name__ == "__main__":
  app.run(main)
