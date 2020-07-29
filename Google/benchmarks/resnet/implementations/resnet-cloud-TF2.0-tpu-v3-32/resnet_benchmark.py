# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Executes Resnet benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import flags
import tensorflow as tf
from tf2_common.utils.flags import core as flags_core
from tf2_common.utils.testing import benchmark_wrappers
from tf2_common.utils.testing.perfzero_benchmark import PerfZeroBenchmark
from tf2_resnet import resnet_ctl_imagenet_main

MIN_TOP_1_ACCURACY = 0.759
MAX_TOP_1_ACCURACY = 0.77

FLAGS = flags.FLAGS


class ResnetBenchmarkReal(PerfZeroBenchmark):
  """Resnet benchmark with real data."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               root_data_dir=None,
               flag_methods=None,
               tpu=None,
               **kwargs):
    self.root_data_dir = root_data_dir
    flag_methods = [resnet_ctl_imagenet_main.define_imagenet_keras_flags]
    super(ResnetBenchmarkReal, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods,
        tpu=tpu)

  def _report_benchmark(self,
                        stats,
                        wall_time_sec,
                        top_1_max=None,
                        top_1_min=None,
                        total_batch_size=None,
                        log_steps=None,
                        warmup=1,
                        start_time_sec=None):
    """Report benchmark results by writing to local protobuf file.

    Args:
      stats: dict returned from keras models with known entries.
      wall_time_sec: the during of the benchmark execution in seconds
      top_1_max: highest passing level for top_1 accuracy.
      top_1_min: lowest passing level for top_1 accuracy.
      total_batch_size: Global batch-size.
      log_steps: How often the log was created for stats['step_timestamp_log'].
      warmup: number of entries in stats['step_timestamp_log'] to ignore.
      start_time_sec: the start time of the program in seconds since epoch.
    """

    metrics = []
    if 'eval_acc' in stats:
      metrics.append({
          'name': 'accuracy_top_1',
          'value': stats['eval_acc'],
          'min_value': top_1_min,
          'max_value': top_1_max
      })
      if 'eval_loss' in stats:
        # FLAGS.report_accuracy_metrics is enabled.
        metrics.append({'name': 'eval_loss', 'value': stats['eval_loss']})

        metrics.append({
            'name': 'top_1_train_accuracy',
            'value': stats['train_acc']
        })
        metrics.append({'name': 'train_loss', 'value': stats['train_loss']})

    if (warmup and 'step_timestamp_log' in stats and
        len(stats['step_timestamp_log']) > warmup + 1):
      # first entry in the time_log is start of step 0. The rest of the
      # entries are the end of each step recorded
      time_log = stats['step_timestamp_log']
      steps_elapsed = time_log[-1].batch_index - time_log[warmup].batch_index
      time_elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
      examples_per_sec = total_batch_size * (steps_elapsed / time_elapsed)
      metrics.append({'name': 'exp_per_second', 'value': examples_per_sec})

    if 'avg_exp_per_second' in stats:
      metrics.append({
          'name': 'avg_exp_per_second',
          'value': stats['avg_exp_per_second']
      })

    if start_time_sec and 'step_timestamp_log' in stats:
      time_log = stats['step_timestamp_log']
      # time_log[0] is recorded at the beginning of the first step.
      startup_time = time_log[0].timestamp - start_time_sec
      metrics.append({'name': 'startup_time', 'value': startup_time})

    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=-1,
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={'flags': flags_str})

  @benchmark_wrappers.enable_runtime_flags
  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = resnet_ctl_imagenet_main.run(flags.FLAGS)
    wall_time_sec = time.time() - start_time_sec

    self._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps,
        start_time_sec=start_time_sec)

  def _set_common_flags(self):
    FLAGS.log_steps = 125
    FLAGS.enable_eager = True
    FLAGS.lr_schedule = 'polynomial'
    FLAGS.report_accuracy_metrics = False
    FLAGS.single_l2_loss_op = True
    FLAGS.enable_checkpoint_and_export = False
    FLAGS.training_dataset_cache = True
    FLAGS.eval_dataset_cache = True
    FLAGS.optimizer = 'LARS'
    FLAGS.weight_decay = 0.0002
    FLAGS.clean = True
    FLAGS.label_smoothing = 0.1
    FLAGS.lars_epsilon = 0.0
    FLAGS.epochs_between_evals = 4
    FLAGS.num_classes = 1000
    FLAGS.enable_device_warmup = True
    FLAGS.device_warmup_steps = 1

  def _set_8gpu_common_flags(self):
    self._set_common_flags()
    FLAGS.data_dir = '/data/imagenet/'
    per_gpu_batch_size = 312
    FLAGS.num_gpus = 8
    FLAGS.batch_size = per_gpu_batch_size * FLAGS.num_gpus
    FLAGS.steps_per_loop = int(1281167 / FLAGS.batch_size + 1)
    FLAGS.dtype = 'fp16'
    FLAGS.base_learning_rate = 9.5
    FLAGS.tf_gpu_thread_mode = 'gpu_private'
    FLAGS.datasets_num_private_threads = 32
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.training_prefetch_batchs = 128
    FLAGS.eval_prefetch_batchs = 192
    FLAGS.train_epochs = 40
    FLAGS.warmup_epochs = 5
    FLAGS.eval_offset_epochs = 3

  def benchmark_8_gpu(self):
    super(ResnetBenchmarkReal, self)._setup()
    self._set_8gpu_common_flags()
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
    self._run_and_report_benchmark()

  def benchmark_short_8_gpu(self):
    super(ResnetBenchmarkReal, self)._setup()
    self._set_8gpu_common_flags()
    FLAGS.model_dir = self._get_model_dir('benchmark_short_8_gpu')
    FLAGS.train_epochs = 2
    FLAGS.skip_eval = True
    self._run_and_report_benchmark()

  def benchmark_synth_short_1_gpu(self):
    super(ResnetBenchmarkReal, self)._setup()
    FLAGS.enable_eager = True
    FLAGS.lr_schedule = 'polynomial'
    FLAGS.report_accuracy_metrics = False
    FLAGS.single_l2_loss_op = True
    FLAGS.enable_checkpoint_and_export = False
    FLAGS.optimizer = 'LARS'
    FLAGS.clean = True
    FLAGS.num_classes = 1000
    FLAGS.enable_device_warmup = False
    FLAGS.use_synthetic_data = True
    per_gpu_batch_size = 312
    FLAGS.num_gpus = 1
    FLAGS.batch_size = per_gpu_batch_size * FLAGS.num_gpus
    FLAGS.steps_per_loop = 1
    FLAGS.dtype = 'fp16'
    FLAGS.distribution_strategy = 'mirrored'
    FLAGS.train_steps = 20
    FLAGS.log_steps = 10
    FLAGS.skip_eval = True
    FLAGS.model_dir = self._get_model_dir('benchmark_synth_short_1_gpu')
    self._run_and_report_benchmark()

  def _set_df_common_flags(self):
    FLAGS.data_dir = 'gs://mlperf-imagenet/imagenet/combined'
    self._set_common_flags()
    FLAGS.batch_size = 4096
    FLAGS.steps_per_loop = int(1281167 / FLAGS.batch_size + 1)
    FLAGS.base_learning_rate = 14.0
    FLAGS.dtype = 'bf16'
    FLAGS.cache_decoded_image = True
    FLAGS.tpu_zone = 'europe-west4-a'
    FLAGS.distribution_strategy = 'tpu'
    FLAGS.train_epochs = 41
    FLAGS.warmup_epochs = 2
    FLAGS.eval_offset_epochs = 0

  def benchmark_4x4_tpu(self):
    super(ResnetBenchmarkReal, self)._setup()
    self._set_df_common_flags()
    FLAGS.model_dir = self._get_model_dir('benchmark_4x4_tpu')
    self._run_and_report_benchmark()

  def benchmark_short_4x4_tpu(self):
    super(ResnetBenchmarkReal, self)._setup()
    self._set_df_common_flags()
    FLAGS.model_dir = self._get_model_dir('benchmark_short_4x4_tpu')
    FLAGS.train_epochs = 2
    FLAGS.skip_eval = True
    self._run_and_report_benchmark()


if __name__ == '__main__':
  tf.compat.v2.test.main()
