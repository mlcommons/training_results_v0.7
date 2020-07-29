# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For loading data into NMT models."""
from __future__ import print_function

import tensorflow.compat.v1 as tf

__all__ = ["get_iterator", "get_infer_iterator"]


# pylint: disable=g-long-lambda,line-too-long
def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 global_batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True,
                 filter_oversized_sequences=False,
                 return_raw=False):
  """Function that returns input dataset."""
  # Total number of examples in src_dataset/tgt_dataset
  if not output_buffer_size:
    output_buffer_size = global_batch_size * 100

  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  # Filter oversized input sequences (542 examples are filtered).
  if filter_oversized_sequences:
    src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt: tf.logical_and(
        tf.size(src) <= src_max_len - 2,
        tf.size(tgt) <= tgt_max_len - 1))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len - 2], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.concat(([tgt_sos_id], src, [src_eos_id]), 0),
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls)
  if return_raw:

    def map_fn(src, tgt_in, tgt_out, src_len, tgt_len):
      """Pad the dataset and emit the bucket id as key."""
      src = tf.pad(
          src, [[0, src_max_len - tf.size(src)]], constant_values=src_eos_id)
      tgt_in = tf.pad(
          tgt_in, [[0, tgt_max_len - tf.size(tgt_in)]],
          constant_values=tgt_eos_id)
      tgt_out = tf.pad(
          tgt_out, [[0, tgt_max_len - tf.size(tgt_out)]],
          constant_values=tgt_eos_id)
      bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      bucket_id = tf.cast(
          tf.minimum(
              num_buckets,
              tf.maximum(src_len // bucket_width, tgt_len // bucket_width)),
          tf.int32)
      return tf.concat([
          src, tgt_in, tgt_out,
          tf.reshape(src_len, [1]),
          tf.reshape(tgt_len, [1]),
          tf.reshape(bucket_id, [1])
      ], 0)

    src_tgt_dataset = src_tgt_dataset.map(
        map_fn, num_parallel_calls=num_parallel_calls)
    return src_tgt_dataset.batch(1024)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  src_tgt_dataset = src_tgt_dataset.cache()
  # TODO(saeta): investigate shuffle_and_repeat.
  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed,
      reshuffle_each_iteration).repeat()

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([src_max_len]),  # src
            tf.TensorShape([tgt_max_len]),  # tgt_input
            tf.TensorShape([tgt_max_len]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0),
        # For TPU, must set drop_remainder to True or batch size will be None
        drop_remainder=True)  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      """Calculate bucket_width by maximum source sequence length."""
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=global_batch_size))
  else:
    batched_dataset = batching_func(src_tgt_dataset)


# Make_one_shot_iterator is not applicable here since we have lookup table.
# Instead return a tf.data.dataset and let TpuEstimator to initialize and make
# iterator out of it.
  batched_dataset = batched_dataset.map(
      lambda src, tgt_in, tgt_out, source_size, tgt_in_size: (
          {"source": src,
           "target_input": tgt_in,
           "target_output": tgt_out,
           "source_sequence_length": source_size,
           "target_sequence_length": tgt_in_size}))
  return batched_dataset


# pylint: disable=g-long-lambda,line-too-long
def get_preprocessed_iterator(dataset_file,
                              batch_size,
                              random_seed,
                              max_seq_len,
                              num_buckets,
                              shard_index,
                              num_shards,
                              num_parallel_calls=100):
  """Get the dataset iterator from preprocessed data."""
  dataset = tf.data.Dataset.list_files(
      dataset_file, shuffle=False).shard(num_shards, shard_index)

  def fetch_dataset(filename):
    dataset = tf.data.FixedLengthRecordDataset(filename,
                                               (max_seq_len * 3 + 3) * 4)
    return dataset

  # TODO(dehao, jsimsa): Investigate why using dataset.interleave is slower
  dataset = dataset.apply(
      tf.data.experimental.parallel_interleave(
          fetch_dataset, cycle_length=num_parallel_calls, sloppy=True))

  def _parse(record):
    record = tf.decode_raw(record, tf.int32)
    r = tf.split(record, [max_seq_len, max_seq_len, max_seq_len, 1, 1, 1])
    return tf.cast(tf.reshape(r[5], []), tf.int64), r[0], r[1], r[2], r[3], r[4]

  shuffle_buffer_size = batch_size * 50
  src_tgt_dataset = dataset.map(
      _parse, num_parallel_calls=shuffle_buffer_size).cache()
  src_tgt_dataset = src_tgt_dataset.shuffle(shuffle_buffer_size, random_seed,
                                            True).repeat()

  if num_buckets > 1:

    def key_func(key, unused_1, unused_2, unused_3, unused_src_len,
                 unused_tgt_len):
      return key

    def reduce_func(unused_key, windowed_data):
      return windowed_data.batch(batch_size, drop_remainder=True)

    batched_dataset = src_tgt_dataset.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  else:
    batched_dataset = src_tgt_dataset.batch(batch_size, drop_remainder=True)

  batched_dataset = batched_dataset.map(
      lambda unused_key, src, tgt_in, tgt_out, source_size, tgt_in_size: ({
          "source": tf.reshape(src, [batch_size, max_seq_len]),
          "target_input": tf.reshape(tgt_in, [batch_size, max_seq_len]),
          "target_output": tf.reshape(tgt_out, [batch_size, max_seq_len]),
          "source_sequence_length": tf.reshape(source_size, [batch_size]),
          "target_sequence_length": tf.reshape(tgt_in_size, [batch_size])
      }),
      # TODO(dehao): tune the magic prefetch buffer size.
      num_parallel_calls=batch_size).prefetch(4)
  return batched_dataset


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       sos,
                       src_max_len=None):
  """Get dataset for inference."""
  # Totol number of examples in src_dataset
  # (3003 examples + 69 padding examples).
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_sos_id = tf.cast(src_vocab_table.lookup(tf.constant(sos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))

  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (tf.concat(
      ([src_sos_id], src, [src_eos_id]), 0), 2 + tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([src_max_len]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0),
        drop_remainder=True)  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_dataset = batched_dataset.map(
      lambda src_ids, src_seq_len: (
          {"source": src_ids,
           "source_sequence_length": src_seq_len}))
  return batched_dataset
