# Copyright 2018 Google. All Rights Reserved.
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
"""Training script for SSD.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import multiprocessing
import sys
import threading
from absl import app
import tensorflow.compat.v1 as tf


from REDACTED.mlp_log import mlp_log
from REDACTED.ssd import coco_metric
from REDACTED.ssd import dataloader
from REDACTED.ssd import ssd_constants
from REDACTED.ssd import ssd_model
from REDACTED.util import train_and_eval_runner
# copybara:strip_begin
from REDACTED.REDACTED.multiprocessing import REDACTEDprocess
# copybara:strip_end

tf.flags.DEFINE_string(
    'resnet_checkpoint',
    '/REDACTED/mb-d/home/tpu-perf-team/ssd_checkpoint/resnet34_bs2048_2',
    'Location of the ResNet checkpoint to use for model '
    'initialization.')
tf.flags.DEFINE_string('hparams', '',
                       'Comma separated k=v pairs of hyperparameters.')
tf.flags.DEFINE_integer(
    'num_shards', default=8, help='Number of shards (TPU cores) for '
    'training.')
tf.flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
tf.flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
tf.flags.DEFINE_integer('eval_samples', 5000, 'The number of samples for '
                        'evaluation.')
tf.flags.DEFINE_integer(
    'iterations_per_loop', 1000, 'Number of iterations per TPU training loop')
tf.flags.DEFINE_string(
    'training_file_pattern',
    'REDACTEDtrain*',
    'Glob for training data files (e.g., COCO train - minival set)')
tf.flags.DEFINE_string(
    'validation_file_pattern',
    'REDACTEDval*',
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
tf.flags.DEFINE_bool(
    'use_fake_data', False,
    'Use fake data to reduce the input preprocessing overhead (for unit tests)')
tf.flags.DEFINE_string(
    'val_json_file',
    'REDACTEDinstances_val2017.json',
    'COCO validation JSON containing golden bounding boxes.')
tf.flags.DEFINE_integer('num_examples_per_epoch', 118287,
                        'Number of examples in one epoch')
tf.flags.DEFINE_integer('num_epochs', 64, 'Number of epochs for training')
tf.flags.DEFINE_multi_integer(
    'input_partition_dims',
    default=None,
    help=('Number of partitions on each dimension of the input. Each TPU core'
          ' processes a partition of the input image in parallel using spatial'
          ' partitioning.'))
tf.flags.DEFINE_bool('run_cocoeval', True, 'Whether to run cocoeval')

FLAGS = tf.flags.FLAGS
_STOP = -1


def construct_run_config(iterations_per_loop):
  """Construct the run config."""

  # Parse hparams
  hparams = ssd_model.default_hparams()
  hparams.parse(FLAGS.hparams)

  return dict(
      hparams.values(),
      num_shards=FLAGS.num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      model_dir=FLAGS.model_dir,
      iterations_per_loop=iterations_per_loop,
      steps_per_epoch=FLAGS.num_examples_per_epoch // FLAGS.train_batch_size,
      eval_samples=FLAGS.eval_samples,
      transpose_input=False if FLAGS.input_partition_dims is not None else True,
      use_spatial_partitioning=True
      if FLAGS.input_partition_dims is not None else False,
  )


# copybara:strip_begin
def REDACTED_predict_post_processing():
  """REDACTED batch-processes the predictions."""
  q_in, q_out = REDACTEDprocess.get_user_data()
  predict_post_processing(q_in, q_out)


# copybara:strip_end


def predict_post_processing(q_in, q_out):
  """Run post-processing on CPU for predictions."""
  coco_gt = coco_metric.create_coco(FLAGS.val_json_file, use_cpp_extension=True)

  current_step, predictions = q_in.get()
  while current_step != _STOP and q_out is not None:
    q_out.put((current_step,
               coco_metric.compute_map(
                   predictions,
                   coco_gt,
                   use_cpp_extension=True,
                   nms_on_tpu=True)))
    current_step, predictions = q_in.get()


def main(argv):
  del argv  # Unused.

  params = construct_run_config(FLAGS.iterations_per_loop)
  mlp_log.mlperf_print(key='cache_clear', value=True)
  mlp_log.mlperf_print(key='init_start', value=None)
  mlp_log.mlperf_print('global_batch_size', FLAGS.train_batch_size)
  mlp_log.mlperf_print('opt_base_learning_rate', params['base_learning_rate'])
  mlp_log.mlperf_print(
      'opt_learning_rate_decay_boundary_epochs',
      [params['first_lr_drop_epoch'], params['second_lr_drop_epoch']])
  mlp_log.mlperf_print('opt_weight_decay', params['weight_decay'])
  mlp_log.mlperf_print(
      'model_bn_span', FLAGS.train_batch_size // FLAGS.num_shards *
      params['distributed_group_size'])
  mlp_log.mlperf_print('max_samples', ssd_constants.NUM_CROP_PASSES)
  mlp_log.mlperf_print('train_samples', FLAGS.num_examples_per_epoch)
  mlp_log.mlperf_print('eval_samples', FLAGS.eval_samples)

  params['batch_size'] = FLAGS.train_batch_size // FLAGS.num_shards
  input_partition_dims = FLAGS.input_partition_dims
  train_steps = FLAGS.num_epochs * FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
  eval_steps = int(math.ceil(FLAGS.eval_samples / FLAGS.eval_batch_size))
  runner = train_and_eval_runner.TrainAndEvalRunner(FLAGS.iterations_per_loop,
                                                    train_steps, eval_steps,
                                                    FLAGS.num_shards)

  train_input_fn = dataloader.SSDInputReader(
      FLAGS.training_file_pattern,
      params['transpose_input'],
      is_training=True,
      use_fake_data=FLAGS.use_fake_data,
      params=params)
  eval_input_fn = dataloader.SSDInputReader(
      FLAGS.validation_file_pattern,
      is_training=False,
      use_fake_data=FLAGS.use_fake_data,
      distributed_eval=True,
      count=eval_steps * FLAGS.eval_batch_size,
      params=params)

  def init_fn():
    tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
        'resnet/': 'resnet%s/' % ssd_constants.RESNET_DEPTH,
    })

  if FLAGS.run_cocoeval:
    # copybara:strip_begin
    q_in, q_out = REDACTEDprocess.get_user_data()
    processes = [
        REDACTEDprocess.Process(target=REDACTED_predict_post_processing) for _ in range(4)
    ]
    # copybara:strip_end_and_replace_begin
    # q_in = multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE)
    # q_out = multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE)
    # processes = [
    #     multiprocessing.Process(
    #         target=predict_post_processing, args=(q_in, q_out))
    #     for _ in range(self.num_multiprocessing_workers)
    # ]
    # copybara:replace_end
    for p in processes:
      p.start()

    def log_eval_results_fn():
      """Print out MLPerf log."""
      result = q_out.get()
      success = False
      while result[0] != _STOP:
        if not success:
          steps_per_epoch = (
              FLAGS.num_examples_per_epoch // FLAGS.train_batch_size)
          epoch = (result[0] + FLAGS.iterations_per_loop) // steps_per_epoch
          mlp_log.mlperf_print(
              'eval_accuracy',
              result[1]['COCO/AP'],
              metadata={'epoch_num': epoch})
          mlp_log.mlperf_print('eval_stop', None, metadata={'epoch_num': epoch})
          if result[1]['COCO/AP'] > ssd_constants.EVAL_TARGET:
            success = True
            mlp_log.mlperf_print(
                'run_stop', None, metadata={'status': 'success'})
        result = q_out.get()
      if not success:
        mlp_log.mlperf_print('run_stop', None, metadata={'status': 'abort'})

    log_eval_result_thread = threading.Thread(target=log_eval_results_fn)
    log_eval_result_thread.start()

  runner.initialize(train_input_fn, eval_input_fn,
                    functools.partial(ssd_model.ssd_model_fn,
                                      params), FLAGS.train_batch_size,
                    FLAGS.eval_batch_size, input_partition_dims, init_fn)
  mlp_log.mlperf_print('init_stop', None)
  mlp_log.mlperf_print('run_start', None)

  def eval_init_fn(cur_step):
    """Executed before every eval."""
    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    epoch = cur_step // steps_per_epoch
    mlp_log.mlperf_print(
        'block_start',
        None,
        metadata={
            'first_epoch_num': epoch,
            'epoch_count': FLAGS.iterations_per_loop // steps_per_epoch
        })
    mlp_log.mlperf_print(
        'eval_start',
        None,
        metadata={
            'epoch_num': epoch + FLAGS.iterations_per_loop // steps_per_epoch
        })

  def eval_finish_fn(cur_step, eval_output, _):
    steps_per_epoch = FLAGS.num_examples_per_epoch // FLAGS.train_batch_size
    epoch = cur_step // steps_per_epoch
    mlp_log.mlperf_print(
        'block_stop',
        None,
        metadata={
            'first_epoch_num': epoch,
            'epoch_count': FLAGS.iterations_per_loop // steps_per_epoch
        })
    if FLAGS.run_cocoeval:
      q_in.put((cur_step, eval_output['detections']))

  runner.train_and_eval(eval_init_fn, eval_finish_fn)

  if FLAGS.run_cocoeval:
    for _ in processes:
      q_in.put((_STOP, None))

    for p in processes:
      try:
        p.join(timeout=10)
      except Exception:  #  pylint: disable=broad-except
        pass

    q_out.put((_STOP, None))
    log_eval_result_thread.join()

    # Clear out all the queues to avoid deadlock.
    while not q_out.empty():
      q_out.get()
    while not q_in.empty():
      q_in.get()


if __name__ == '__main__':
  # copybara:strip_begin
  user_data = (multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE),
               multiprocessing.Queue(maxsize=ssd_constants.QUEUE_SIZE))
  in_compile_test = False
  for arg in sys.argv:
    if arg == '--xla_jf_exit_process_on_compilation_success=true':
      in_compile_test = True
      break
  if in_compile_test:
    # Exiting from XLA's C extension skips REDACTEDprocess's multiprocessing clean
    # up. Don't use REDACTED process when xla is in compilation only mode.
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
  else:
    with REDACTEDprocess.main_handler(user_data=user_data):
      tf.logging.set_verbosity(tf.logging.INFO)
      app.run(main)
  # copybara:strip_end
  # copybara:insert tf.logging.set_verbosity(tf.logging.INFO)
  # copybara:insert app.run(main)
