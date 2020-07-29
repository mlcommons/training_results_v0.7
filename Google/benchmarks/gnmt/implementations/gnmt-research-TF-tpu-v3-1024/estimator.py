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
"""Estimator functions supporting running on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tensorflow.compat.v1 as tf
from REDACTED.tensorflow.contrib import learn as contrib_learn
from REDACTED.tensorflow.contrib import tpu as contrib_tpu

from REDACTED.nmt import low_level_runner
from REDACTED.nmt import metric
from REDACTED.nmt import model
from REDACTED.nmt.utils import iterator_utils
from REDACTED.nmt.utils import vocab_utils


def make_model_fn(hparams):
  """Construct a GNMT model function for training."""

  def _model_fn(features, labels, mode, params):
    """Model function."""
    del labels, params
    # Create a GNMT model for training.
    gnmt_model = model.BaseModel(hparams, mode=mode, features=features)
    if mode == contrib_learn.ModeKeys.INFER:
      predicted_ids = gnmt_model.predicted_ids
      # make sure outputs is of shape [batch_size, time] or [beam_width,
      # batch_size, time] when using beam search.
      predicted_ids = tf.transpose(predicted_ids, [2, 1, 0])
      # Get the top predictions from beam search.
      predicted_ids = tf.gather_nd(predicted_ids, [0])
      predictions = {"predictions": predicted_ids}
      return contrib_tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

    elif mode == contrib_learn.ModeKeys.TRAIN:
      loss = tf.zeros([], dtype=tf.float32)
      train_op = gnmt_model.update

    else:
      raise ValueError("Unknown mode in model_fn: %s" % mode)

    if hparams.use_tpu:
      return contrib_tpu.TPUEstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    else:
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  return _model_fn


def make_input_fn(hparams, mode):
  """Construct a input function for training."""

  def _input_fn(params):
    """Input function."""
    if mode == contrib_learn.ModeKeys.TRAIN:
      src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
      tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    else:
      src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
      tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    src_vocab_file = hparams.src_vocab_file
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file)

    if mode == contrib_learn.ModeKeys.TRAIN:
      if "context" in params:
        batch_size = params["batch_size"]
        global_batch_size = batch_size
        num_hosts = params["context"].num_hosts
        # TODO(dehao): update to use current_host once available in API.
        current_host = params["context"].current_input_fn_deployment()[1]
      else:
        if "dataset_index" in params:
          current_host = params["dataset_index"]
          num_hosts = params["dataset_num_shards"]
          batch_size = params["batch_size"]
          global_batch_size = hparams.batch_size
        else:
          num_hosts = 1
          current_host = 0
          batch_size = hparams.batch_size
          global_batch_size = batch_size
      if not hparams.use_preprocessed_data:
        src_dataset = tf.data.TextLineDataset(src_file)
        tgt_dataset = tf.data.TextLineDataset(tgt_file)
        return iterator_utils.get_iterator(
            src_dataset,
            tgt_dataset,
            src_vocab_table,
            tgt_vocab_table,
            batch_size=batch_size,
            global_batch_size=global_batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            output_buffer_size=None,
            skip_count=None,
            num_shards=num_hosts,
            shard_index=current_host,
            reshuffle_each_iteration=True,
            filter_oversized_sequences=True)
      else:
        return iterator_utils.get_preprocessed_iterator(
            hparams.train_prefix + "*",
            batch_size=batch_size,
            random_seed=hparams.random_seed,
            max_seq_len=hparams.src_max_len,
            num_buckets=hparams.num_buckets,
            shard_index=current_host,
            num_shards=num_hosts)
    else:
      if "dataset_index" in params:
        current_host = params["dataset_index"]
        num_hosts = params["dataset_num_shards"]
      else:
        num_hosts = 1
        current_host = 0
      if "infer_batch_size" in params:
        batch_size = params["infer_batch_size"]
      else:
        batch_size = hparams.infer_batch_size
      src_dataset = tf.data.TextLineDataset(src_file)
      src_dataset = src_dataset.repeat().batch(
          hparams.infer_batch_size // num_hosts).shard(num_hosts,
                                                       current_host).unbatch()
      return iterator_utils.get_infer_iterator(
          src_dataset,
          src_vocab_table,
          batch_size=batch_size,
          eos=hparams.eos,
          sos=hparams.sos,
          src_max_len=hparams.src_max_len_infer)

  def _synthetic_input_fn(params):
    """Fake inputs for debugging and benchmarking."""
    del params
    batch_size = hparams.batch_size
    src_max_len = hparams.src_max_len
    tgt_max_len = hparams.tgt_max_len
    features = {
        "source":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=1,
                shape=(batch_size, src_max_len)),
        "target_input":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=2,
                shape=(batch_size, tgt_max_len)),
        "target_output":
            tf.random_uniform(
                dtype=tf.int32,
                minval=1,
                maxval=10,
                seed=3,
                shape=(batch_size, tgt_max_len)),
        "source_sequence_length":
            tf.constant([src_max_len] * batch_size),
        "target_sequence_length":
            tf.constant([tgt_max_len] * batch_size)
    }
    return features

  if hparams.use_synthetic_data and mode == contrib_learn.ModeKeys.TRAIN:
    return _synthetic_input_fn
  else:
    return _input_fn


def _get_tgt_sos_eos_id(hparams):
  with tf.Session() as sess:
    _, tgt_vocab_table = vocab_utils.create_vocab_tables(
        hparams.src_vocab_file)
    tgt_sos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
    tgt_eos_id = tf.cast(
        tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)
    sess.run(tf.tables_initializer())
    tgt_sos_id = sess.run(tgt_sos_id, {})
    tgt_eos_id = sess.run(tgt_eos_id, {})
    return tgt_sos_id, tgt_eos_id


def create_train_runner(hparams, eval_steps=0):
  steps_per_epoch = int(hparams.num_examples_per_epoch/hparams.batch_size)
  return low_level_runner.LowLevelRunner(
      train_iterations=steps_per_epoch,
      eval_steps=eval_steps,
      hparams=hparams,
      per_host_v1=True)


def create_eval_runner(hparams):
  hparams.tgt_sos_id, hparams.tgt_eos_id = 1, 2
  eval_steps = int(math.ceil(
      hparams.examples_to_infer / hparams.infer_batch_size))
  return low_level_runner.LowLevelRunner(
      eval_steps=eval_steps, hparams=hparams, train_iterations=0)


def create_eval_runner_and_build_graph(hparams, model_fn):
  runner = create_eval_runner(hparams)
  input_fn = make_input_fn(hparams, contrib_learn.ModeKeys.INFER)
  params = {
      "infer_batch_size": int(hparams.infer_batch_size / hparams.num_shards)
  }
  runner.initialize(None, input_fn, params)
  runner.build_model(model_fn, params)
  return runner


def train_fn(hparams):
  """Train function."""
  hparams.tgt_sos_id, hparams.tgt_eos_id = _get_tgt_sos_eos_id(hparams)
  model_fn = make_model_fn(hparams)

  runner = create_train_runner(hparams)
  input_fn = make_input_fn(hparams, contrib_learn.ModeKeys.TRAIN)
  runner.initialize(input_fn, None, {})
  runner.build_model(model_fn, {})
  runner.train(0, hparams.num_train_steps)
  return 0.0


def train_and_eval_with_low_level_api(hparams):
  """Train and evaluation function."""
  # pylint: disable=protected-access
  hparams.tgt_sos_id, hparams.tgt_eos_id = 1, 2
  model_fn = make_model_fn(hparams)
  eval_steps = int(
      math.ceil(hparams.examples_to_infer / hparams.infer_batch_size))
  runner = create_train_runner(hparams, eval_steps)
  train_input_fn = make_input_fn(hparams, contrib_learn.ModeKeys.TRAIN)
  eval_input_fn = make_input_fn(hparams, contrib_learn.ModeKeys.INFER)
  params = {
      "infer_batch_size": int(hparams.infer_batch_size / hparams.num_shards)
  }
  runner.initialize(train_input_fn, eval_input_fn, params)
  runner.build_model(model_fn, params)

  return runner.train_and_predict()


def eval_fn(hparams):
  """Inference function."""
  hparams.tgt_sos_id, hparams.tgt_eos_id = _get_tgt_sos_eos_id(hparams)
  model_fn = make_model_fn(hparams)
  eval_runner = create_eval_runner_and_build_graph(hparams, model_fn)
  predictions = list(eval_runner.predict())
  checkpoint_path = tf.train.latest_checkpoint(hparams.out_dir)
  current_step = int(os.path.basename(checkpoint_path).split("-")[1])
  return metric.get_metric(hparams, predictions, current_step)
