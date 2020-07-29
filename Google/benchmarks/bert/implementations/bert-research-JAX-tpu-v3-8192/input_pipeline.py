"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow.compat.v1 as tf
from REDACTED.tensorflow.python.data import experimental as experimental_data


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4,
                     global_input_size=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params, ctx=None):
    """The actual input function."""
    input_context = ctx
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    if input_context:
      num_hosts = input_context.num_input_pipelines
      host_index = input_context.input_pipeline_id
    else:
      num_hosts = 1
      host_index = 0

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      input_length = len(input_files)
      if num_hosts > 1:
        input_length = int(math.ceil(input_length / num_hosts))
        tf.logging.info(
            "Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d"
            % (host_index, num_hosts))
        d = d.shard(num_hosts, host_index)

      d = d.shuffle(buffer_size=input_length)
      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, input_length)

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          experimental_data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=1000)
      d = d.repeat()
    else:
      d = tf.data.TFRecordDataset(input_files)
      tf.logging.info(
          "Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d" %
          (host_index, num_hosts))

      if global_input_size:
        d = d.take(global_input_size)
      d = d.shard(num_hosts, host_index)
      global_bs = num_hosts * batch_size
      if global_input_size and global_input_size % global_bs != 0:
        # Round up the number of eval steps global_input_size / global_bs.
        num_dataset_per_shard = max(
            1, int(math.ceil(global_input_size / global_bs) * batch_size))

        def _float_feature(values):
          """Returns a float_list from a float / double."""
          return tf.train.Feature(
              float_list=tf.train.FloatList(value=list(values)))

        def _int64_feature(values):
          """Returns an int64_list from a bool / enum / int / uint."""
          return tf.train.Feature(
              int64_list=tf.train.Int64List(value=list(values)))
        # Since masked_lm_weights is provided as 0s, the score coming from
        # the eval of the padding will be multipled with 0.
        # This won't affect the average either, as the padding weight will be 0.
        padded_dataset = tf.data.Dataset.from_tensors(
            tf.constant(
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "input_ids":
                                _int64_feature([0]*max_seq_length),
                            "input_mask":
                                _int64_feature([0]*max_seq_length),
                            "segment_ids":
                                _int64_feature([0]*max_seq_length),
                            "masked_lm_positions":
                                _int64_feature([0]*max_predictions_per_seq),
                            "masked_lm_ids":
                                _int64_feature([0]*max_predictions_per_seq),
                            "masked_lm_weights":
                                _float_feature([0]*max_predictions_per_seq),
                            "next_sentence_labels":
                                _int64_feature([0]),
                        })).SerializeToString(),
                dtype=tf.string)).repeat(num_dataset_per_shard)

        d = d.concatenate(padded_dataset).take(num_dataset_per_shard)
        tf.logging.info(
            "Padding the dataset: input_pipeline_id=%d padded_size=%d" %
            (host_index,
             num_dataset_per_shard - global_input_size / num_hosts))
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example

