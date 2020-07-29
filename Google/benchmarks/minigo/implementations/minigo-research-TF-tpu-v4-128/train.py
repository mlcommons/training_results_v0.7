# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a network.

Usage:
  BOARD_SIZE=19 python train.py tfrecord1 tfrecord2 tfrecord3
"""

import logging
import math

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.minigo import dual_net
from REDACTED.minigo import preprocessing
from REDACTED.minigo import utils

# See www.moderndescartes.com/essays/shuffle_viz for discussion on sizing
flags.DEFINE_integer('shuffle_buffer_size', 2000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_boolean('shuffle_examples', True,
                     'Whether to shuffle training examples.')

flags.DEFINE_integer(
    'steps_to_train', None,
    'Number of training steps to take. If not set, iterates '
    'once over training data.')

flags.DEFINE_integer(
    'num_examples', None,
    'Total number of input examples. This is only used if '
    'steps_to_train is not set. Requires that filter_amount '
    'is 1.0.')

flags.DEFINE_integer('window_size', 500000,
                     'Number of games to include in the window')

flags.DEFINE_float(
    'filter_amount', 1.0, 'Fraction of positions to filter from golden chunks,'
    'default, 1.0 (no filter)')

flags.DEFINE_string('export_path', None,
                    'Where to export the model after training.')

flags.DEFINE_string('train_data_path', None, 'Path to training examples.')

flags.DEFINE_bool('freeze', False,
                  'Whether to freeze the graph at the end of training.')

flags.DEFINE_boolean('use_trt', False,
                     'True to write a GraphDef that uses the TRT runtime')
flags.DEFINE_integer('trt_max_batch_size', None, 'Maximum TRT batch size')
flags.DEFINE_string('trt_precision', 'fp32',
                    'Precision for TRT runtime: fp16, fp32 or int8')
flags.register_multi_flags_validator(
    ['use_trt', 'trt_max_batch_size'],
    lambda flags: not flags['use_trt'] or flags['trt_max_batch_size'],
    'trt_max_batch_size must be set if use_trt is true')


@flags.multi_flags_validator(
    ['num_examples', 'steps_to_train', 'filter_amount'],
    '`num_examples` requires `steps_to_train==0` and `filter_amount==1.0`')
def _example_flags_validator(flags_dict):
  if not flags_dict['num_examples']:
    return True
  return not flags_dict['steps_to_train'] and flags_dict['filter_amount'] == 1.0


# From dual_net.py
flags.declare_key_flag('work_dir')
flags.declare_key_flag('train_batch_size')
flags.declare_key_flag('num_tpu_cores')
flags.declare_key_flag('use_tpu')
flags.declare_key_flag('input_layout')

FLAGS = flags.FLAGS


class EchoStepCounterHook(tf.train.StepCounterHook):
  """A hook that logs steps per second."""

  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    s_per_sec = elapsed_steps / elapsed_time
    logging.info('{}: {:.3f} steps per second'.format(global_step, s_per_sec))
    super()._log_and_record(elapsed_steps, elapsed_time, global_step)  # pylint: disable=missing-super-argument


def compute_update_ratio(weight_tensors, before_weights, after_weights):
  """Compute the ratio of gradient norm to weight norm."""
  deltas = [
      after - before for after, before in zip(after_weights, before_weights)
  ]
  delta_norms = [np.linalg.norm(d.ravel()) for d in deltas]
  weight_norms = [np.linalg.norm(w.ravel()) for w in before_weights]
  ratios = [d / w for d, w in zip(delta_norms, weight_norms)]
  all_summaries = [
      tf.Summary.Value(tag='update_ratios/' + tensor.name, simple_value=ratio)
      for tensor, ratio in zip(weight_tensors, ratios)
  ]
  return tf.Summary(value=all_summaries)


class UpdateRatioSessionHook(tf.train.SessionRunHook):
  """A hook that computes ||grad|| / ||weights|| (using frobenius norm)."""

  def __init__(self, output_dir, every_n_steps=1000):
    self.output_dir = output_dir
    self.every_n_steps = every_n_steps
    self.before_weights = None
    self.file_writer = None
    self.weight_tensors = None
    self.global_step = None

  def begin(self):
    """Called once before using the session."""
    # These calls only works because the SessionRunHook api guarantees this
    # will get called within a graph context containing our model graph.

    self.file_writer = tf.summary.FileWriterCache.get(self.output_dir)
    self.weight_tensors = tf.trainable_variables()
    self.global_step = tf.train.get_or_create_global_step()

  def before_run(self, run_context):
    """Called before each call to run()."""
    global_step = run_context.session.run(self.global_step)
    if global_step % self.every_n_steps == 0:
      self.before_weights = run_context.session.run(self.weight_tensors)

  def after_run(self, run_context, unused_run_values):
    """Called after each call to run()."""
    global_step = run_context.session.run(self.global_step)
    if self.before_weights is not None:
      after_weights = run_context.session.run(self.weight_tensors)
      weight_update_summaries = compute_update_ratio(self.weight_tensors,
                                                     self.before_weights,
                                                     after_weights)
      self.file_writer.add_summary(weight_update_summaries, global_step)
      self.before_weights = None


def train(*tf_records):
  """Train on examples.

  Args:
    *tf_records: records to train on.
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.compat.v1.reset_default_graph()
  estimator = dual_net.get_estimator()

  effective_batch_size = FLAGS.train_batch_size
  if FLAGS.use_tpu:
    effective_batch_size *= FLAGS.num_tpu_cores

  if FLAGS.use_tpu:

    def _input_fn(params):
      return preprocessing.get_tpu_input_tensors(
          params['batch_size'],
          params['input_layout'],
          tf_records,
          filter_amount=FLAGS.filter_amount,
          shuffle_examples=FLAGS.shuffle_examples,
          shuffle_buffer_size=FLAGS.shuffle_buffer_size,
          random_rotation=True)

    # Hooks are broken with TPUestimator at the moment.
    hooks = []
  else:

    def _input_fn():
      return preprocessing.get_input_tensors(
          FLAGS.train_batch_size,
          FLAGS.input_layout,
          tf_records,
          filter_amount=FLAGS.filter_amount,
          shuffle_examples=FLAGS.shuffle_examples,
          shuffle_buffer_size=FLAGS.shuffle_buffer_size,
          random_rotation=True)

    hooks = [
        UpdateRatioSessionHook(FLAGS.work_dir),
        EchoStepCounterHook(output_dir=FLAGS.work_dir)
    ]

  steps = FLAGS.steps_to_train
  if not steps and FLAGS.num_examples:
    batch_size = FLAGS.train_batch_size
    if FLAGS.use_tpu:
      batch_size *= FLAGS.num_tpu_cores
    steps = math.floor(FLAGS.num_examples / batch_size)

  logging.info('Training, steps = %s, batch = %s -> %s examples', steps or '?',
               effective_batch_size,
               (steps * effective_batch_size) if steps else '?')

  try:
    estimator.train(_input_fn, steps=steps, hooks=hooks)
  except:
    raise


def main(unused_args):
  """Train on examples and export the updated model weights."""
  tf_records = tf.gfile.Glob(FLAGS.train_data_path)
  logging.info('Training on %s records: %s to %s', len(tf_records),
               tf_records[0], tf_records[-1])
  with utils.logged_timer('Training'):
    train(*tf_records)
  if FLAGS.freeze:
    if FLAGS.use_tpu:
      dual_net.freeze_graph_tpu(FLAGS.export_path)
    else:
      dual_net.freeze_graph(FLAGS.export_path, FLAGS.use_trt,
                            FLAGS.trt_max_batch_size, FLAGS.trt_precision)
  if FLAGS.export_path:
    dual_net.export_model(FLAGS.export_path)


if __name__ == '__main__':
  app.run(main)
