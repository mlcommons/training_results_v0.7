# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Build input pipelines that span TPU pods for optimal performance.

It's common to batch sequences according to their length. Unfortunately, a
naive scaling of such an input pipeline across a pod will result in each host
choosing the sequence length bucket independently. Concretely, host A may select
sequences of a short length, while host B may select sequences of a very long
length. Because every step involves a blocking all-reduce phase, host A must
wait for host B.

The input pipeline designed within synchronizes the hosts such that they all
select a sequence length bucket of the same length, resulting in up to 50%
performance improvements across large TPU pod slices.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow.compat.v1 as tf
from REDACTED.tensorflow.python.data.ops import multi_device_iterator_ops
from REDACTED.tensorflow_models.mlperf.models.rough.nmt import low_level_runner
from REDACTED.tensorflow_models.mlperf.models.rough.nmt.utils import vocab_utils


class DistributedPipeline(tf.train.SessionRunHook):
  """DistributedPipeline encapsulates constructing the distributed pipeline.

  We use a class because we need to construct the pipeline in a graph managed
  by [TPU]Estimator. As a result, we cannot pre-construct it using a normal
  function, as Estimator wants to manage the graph itself.

  We use a class because we need to capture the initializer and pass it to the
  train call to TPUEstimator while simultaneously passing ourselves as the input
  function.
  """

  def __init__(self, hparams, num_hosts):
    """Constructs a DistributedPipeline.

    Args:
      hparams: The hparams object for this model.
      num_hosts: The number of hosts in the slice of the TPU pod.

    Throws:
      ValueError: If the passed values are invalid.
    """
    self._hparams = hparams
    self._num_hosts = num_hosts
    self._iterator = None
    self._outputs = None
    global_batch_size = hparams.batch_size
    if global_batch_size % num_hosts != 0:
      raise ValueError(
          "global_batch_size (%s) must be a multiple of num_hosts (%s)" %
          (global_batch_size, num_hosts))

  def after_create_session(self, session, coord):
    del coord
    start = time.time()
    session.run(self._iterator.initializer)
    tf.logging.info("Initialized multi-host dataset iterators in %d seconds",
                    time.time() - start)

  def iterator(self):
    return self._iterator

  def __call__(self, params):
    if not self._outputs:
      self._iterator = _make_distributed_pipeline(self._hparams,
                                                  self._num_hosts)
      self._outputs = self._iterator.get_next()

    if "context" in params:
      current_host = params["context"].current_input_fn_deployment()[1]
    elif "dataset_index" in params:
      current_host = params["dataset_index"]
    else:
      raise ValueError('Expect "context" or "dataset_index" in params.')

    return self._outputs[current_host]


def _make_distributed_pipeline(hparams, num_hosts):
  """Makes the distributed input pipeline.

  make_distributed_pipeline must be used in the PER_HOST_V1 configuration.

  Note: we return both the input function and the hook because
  MultiDeviceIterator is not compatible with Estimator / TPUEstimator.

  Args:
    hparams: The hyperparameters to use.
    num_hosts: The number of hosts we're running across.

  Returns:
    A MultiDeviceIterator.
  """
  # TODO: Merge with the original copy in iterator_utils.py.
  # pylint: disable=g-long-lambda,line-too-long
  global_batch_size = hparams.batch_size

  if global_batch_size % num_hosts != 0:
    raise ValueError(
        "global_batch_size (%s) must be a multiple of num_hosts (%s)" %
        (global_batch_size, num_hosts))

  # Optionally choose from `choose_buckets` buckets simultaneously.
  if hparams.choose_buckets:
    window_batch_size = int(global_batch_size / hparams.choose_buckets)
  else:
    window_batch_size = global_batch_size

  per_host_batch_size = global_batch_size / num_hosts

  output_buffer_size = global_batch_size * 50

  resolver = low_level_runner.get_resolver(hparams)
  if resolver:
    job_name = resolver.get_job_name() or hparams.tpu_job_name or "tpu_worker"
  if hparams.master == "local":
    job_name = "localhost"

  with tf.device("/job:%s/task:0/cpu:0" % job_name):
    # From estimator.py
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file)
    src_dataset = tf.data.TextLineDataset(src_file).prefetch(output_buffer_size)
    tgt_dataset = tf.data.TextLineDataset(tgt_file).prefetch(output_buffer_size)

    # Define local variables that are parameters in iterator_utils.make_input_fn
    sos = hparams.sos
    eos = hparams.eos
    random_seed = hparams.random_seed
    num_buckets = hparams.num_buckets
    src_max_len = hparams.src_max_len
    tgt_max_len = hparams.tgt_max_len
    num_parallel_calls = 100  # constant in iterator_utils.py
    skip_count = None  # constant in estimator.py
    reshuffle_each_iteration = True  # constant in estimator.py
    filter_oversized_sequences = True  # constant in estimator.py

    # From iterator_utils.py
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    if skip_count is not None:
      src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    def map_fn_1(src, tgt):
      src = tf.string_split([src]).values
      tgt = tf.string_split([tgt]).values
      src_size = tf.size(src)
      tgt_size = tf.size(tgt)
      size_ok_bool = tf.logical_and(src_size > 0, tgt_size > 0)
      if filter_oversized_sequences:
        oversized = tf.logical_and(src_size < src_max_len,
                                   tgt_size < tgt_max_len)
        size_ok_bool = tf.logical_and(size_ok_bool, oversized)

      if src_max_len:
        src = src[:src_max_len]
      if tgt_max_len:
        tgt = tgt[:tgt_max_len]
      return (src, tgt, size_ok_bool)

    src_tgt_bool_dataset = src_tgt_dataset.map(
        map_fn_1, num_parallel_calls=num_parallel_calls)
    src_tgt_bool_dataset = src_tgt_bool_dataset.filter(
        lambda src, tgt, filter_bool: filter_bool)

    def map_fn_2(src, tgt, unused_filter_bool):
      src = tf.cast(src_vocab_table.lookup(src), tf.int32)
      tgt = tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)

      # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
      tgt_in = tf.concat(([tgt_sos_id], tgt), 0)
      tgt_out = tf.concat((tgt, [tgt_eos_id]), 0)

      # Add in sequence lengths.
      src_len = tf.size(src)
      tgt_len = tf.size(tgt_in)
      return src, tgt_in, tgt_out, src_len, tgt_len

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_bool_dataset.map(
        map_fn_2, num_parallel_calls=num_parallel_calls)

    def map_fn_3(src, tgt_in, tgt_out, src_len, tgt_len):  # pylint: disable=missing-docstring
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

      return tf.to_int64(tf.minimum(
          num_buckets, bucket_id)), src, tgt_in, tgt_out, src_len, tgt_len

    src_tgt_dataset = src_tgt_dataset.map(
        map_fn_3, num_parallel_calls=num_parallel_calls)

    src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.cache()
    src_tgt_dataset = src_tgt_dataset.shuffle(
        output_buffer_size, random_seed, reshuffle_each_iteration).repeat()

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
      return x.padded_batch(
          window_batch_size,
          # The first three entries are the source and target line rows;
          # these have unknown-length vectors.  The last two entries are
          # the source and target row sizes; these are scalars.
          padded_shapes=(
              tf.TensorShape([]),  # key
              tf.TensorShape([src_max_len]),  # src
              tf.TensorShape([tgt_max_len]),  # tgt_input
              tf.TensorShape([tgt_max_len]),  # tgt_output
              tf.TensorShape([]),  # src_len
              tf.TensorShape([])),  # tgt_len
          # Pad the source and target sequences with eos tokens.
          # (Though notice we don't generally need to do this since
          # later on we will be masking out calculations past the true sequence.
          padding_values=(
              tf.to_int64(0),  # key
              src_eos_id,  # src
              tgt_eos_id,  # tgt_input
              tgt_eos_id,  # tgt_output
              0,  # src_len -- unused
              0),
          # For TPU, must set drop_remainder to True or batch size will be None
          drop_remainder=True)  # tgt_len -- unused

    def key_func(key, unused_1, unused_2, unused_3, unused_4, unused_5):
      return key

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    if num_buckets > 1:
      batched_dataset = src_tgt_dataset.apply(
          tf.data.experimental.group_by_window(
              key_func=key_func,
              reduce_func=reduce_func,
              window_size=window_batch_size))
    else:
      batched_dataset = batching_func(src_tgt_dataset)

    batched_dataset = batched_dataset.map(
        lambda unused_key, src, tgt_in, tgt_out, source_size, tgt_in_size: (
            {"source": src,
             "target_input": tgt_in,
             "target_output": tgt_out,
             "source_sequence_length": source_size,
             "target_sequence_length": tgt_in_size}))

    re_batched_dataset = batched_dataset.unbatch().batch(
        int(per_host_batch_size), drop_remainder=True)

    output_devices = [
        "/job:%s/task:%d/cpu:0" % (job_name, i) for i in range(num_hosts)
    ]

    options = tf.data.Options()
    options.experimental_optimization.filter_fusion = True
    options.experimental_optimization.map_and_filter_fusion = True
    re_batched_dataset = re_batched_dataset.with_options(options)

    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset=re_batched_dataset,
        devices=output_devices,
        max_buffer_size=10,
        prefetch_buffer_size=10,
        source_device=("/job:%s/task:0/cpu:0" % job_name))

    return multi_device_iterator
