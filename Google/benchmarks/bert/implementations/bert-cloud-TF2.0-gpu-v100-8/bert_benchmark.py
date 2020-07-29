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
"""Executes Bert benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
from absl import logging
import numpy as np
import math
import tensorflow.compat.v2 as tf
from REDACTED.tf2_bert.bert import common_flags
from REDACTED.tf2_bert.bert import configs
from REDACTED.tf2_bert.bert import run_pretraining
from REDACTED.tf2_common.modeling import performance
from REDACTED.tf2_common.utils.flags import core as flags_core
from REDACTED.tf2_common.utils.misc import distribution_utils
from REDACTED.tf2_common.utils.testing.perfzero_benchmark import PerfZeroBenchmark

FLAGS = flags.FLAGS


class BenchmarkTimerCallback(tf.keras.callbacks.Callback):
  """Callback that records time it takes to run each batch."""

  def __init__(self, num_batches_to_skip=10):
    super(BenchmarkTimerCallback, self).__init__()
    self.batch_start_times = {}
    self.batch_stop_times = {}

  def on_batch_begin(self, batch, logs=None):
    self.batch_start_times[batch] = time.time()

  def on_batch_end(self, batch, logs=None):
    # If there are multiple steps_per_loop, the end batch index will not be the
    # same as the starting index. Use the last starting index instead.
    if batch not in self.batch_start_times:
      batch = max(self.batch_start_times.keys())

    self.batch_stop_times[batch] = time.time()

  def get_examples_per_sec(self, batch_size, num_batches_to_skip=1):
    batch_durations = []
    for batch in self.batch_start_times:
      if batch in self.batch_stop_times and batch >= num_batches_to_skip:
        batch_durations.append(self.batch_stop_times[batch] -
                               self.batch_start_times[batch])
    if not batch_durations:
      logging.error("No batch durations found.")
      return 0.0
    return batch_size / np.mean(batch_durations)

  def get_startup_time(self, program_start_time):
    return self.batch_start_times[0] - program_start_time


class BertClassifyBenchmarkReal(PerfZeroBenchmark):
  """Bert benchmark with real data."""

  def __init__(self,
               output_dir=None,
               default_flags=None,
               root_data_dir=None,
               flag_methods=None,
               tpu=None,
               **kwargs):
    self.root_data_dir = root_data_dir

    # flag_methods = [resnet_imagenet_main.define_imagenet_keras_flags]
    super(BertClassifyBenchmarkReal, self).__init__(
        output_dir=output_dir,
        default_flags=default_flags,
        flag_methods=flag_methods,
        tpu=tpu)

  def _setup(self):
    """Sets up and resets flags before each test."""
    super(BertClassifyBenchmarkReal, self)._setup()
    self.timer_callback = BenchmarkTimerCallback()
    FLAGS.max_seq_length = 512
    FLAGS.max_predictions_per_seq = 76
    FLAGS.input_files = "REDACTEDpart-*"
    FLAGS.num_train_epochs = 1
    FLAGS.optimizer_type = "lamb"
    FLAGS.explicit_allreduce = False
    FLAGS.allreduce_bytes_per_pack = 0
    FLAGS.do_eval = True
    FLAGS.num_eval_samples = 10000
    FLAGS.device_warmup = True
    FLAGS.stop_threshold = 0.712

    # GPU (w/ gradient accumulation) and TPU shares hyperparameters

    # num_steps_per_epoch is used to decay learning rate linearly for steps
    FLAGS.num_steps_per_epoch = 8103
    FLAGS.epsilon = 1e-6
    FLAGS.beta_1 = 0.9
    FLAGS.beta_2 = 0.999
    FLAGS.learning_rate = 0.0004
    FLAGS.warmup_steps = 0
    FLAGS.scale_loss = True

  def _run_and_report_benchmark(self, force_gpu_memory_alloc=True):
    if force_gpu_memory_alloc:
      # force GPU memory allocation, so we always take the same amount of
      # GPU memory as running in Cloud (see b/151435951)
      gpus = tf.config.experimental.list_physical_devices("GPU")
      if gpus:
        try:
          for gpu_id in range(0, len(gpus)):
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_id], [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=14700)
                ])
          logical_gpus = tf.config.experimental.list_logical_devices("GPU")
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
          # Virtual devices must be set before GPUs have been initialized
          print(e)

    if FLAGS.tpu:
      strategy = distribution_utils.get_distribution_strategy(
          distribution_strategy=FLAGS.distribution_strategy,
          tpu_address=FLAGS.tpu,
          tpu_zone="europe-west4-a")
    else:
      strategy = distribution_utils.get_distribution_strategy(
          distribution_strategy=FLAGS.distribution_strategy,
          all_reduce_alg=FLAGS.all_reduce_alg,
          num_gpus=FLAGS.num_gpus)

    start_time_sec = time.time()
    run_pretraining.run_bert_pretrain(strategy, [self.timer_callback])
    wall_time_sec = time.time() - start_time_sec

    metrics = []
    if self.timer_callback:
      metrics.append({
          "name":
              "exp_per_second",
          "value":
              self.timer_callback.get_examples_per_sec(FLAGS.train_batch_size *
                                                       FLAGS.steps_per_loop)
      })
    else:
      logging.error(
          "exp_per_second not calculated because timer_callback is missing")
      metrics.append({
          "name": "exp_per_second",
          "value": 0.0,
      })

    flags_str = flags_core.get_nondefault_flags_as_str()
    self.report_benchmark(
        iters=-1,
        wall_time=wall_time_sec,
        metrics=metrics,
        extras={"flags": flags_str})

  def _set_8gpu_common(self, benchmark_name):
    FLAGS.distribution_strategy = "mirrored"
    FLAGS.all_reduce_alg = "nccl"
    FLAGS.dtype = "fp16"
    FLAGS.loss_scale = "dynamic"
    FLAGS.num_gpus = 8
    FLAGS.train_batch_size = 440
    FLAGS.steps_per_loop = math.ceil(500000.0 / FLAGS.train_batch_size)
    FLAGS.steps_before_eval_start = math.ceil(3000000.0 /
                                              FLAGS.train_batch_size)
    FLAGS.steps_between_eval = FLAGS.steps_per_loop
    FLAGS.eval_batch_size = 48
    FLAGS.num_accumulation_steps = 11
    # To improve kernel launching across multiple GPUs and improve step time
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    base_data_dir = "REDACTEDpublic/bert_mlperf_data/"
    FLAGS.bert_config_file = base_data_dir + "bert_config.json"
    FLAGS.model_dir = self._get_model_dir(benchmark_name)
    FLAGS.init_checkpoint = base_data_dir + "model.ckpt-28252"

  # Convergence test with same params as reference MLPerf
  def benchmark_8_gpu(self):
    self._setup()
    self._set_8gpu_common("benchmark_8_gpu")
    FLAGS.stop_steps = 8103
    self._run_and_report_benchmark()

  # Short throughput test
  def benchmark_short_8_gpu(self):
    self._setup()
    self._set_8gpu_common("benchmark_short_8_gpu")
    FLAGS.init_checkpoint = None
    FLAGS.steps_per_loop = 200
    FLAGS.steps_between_eval = 200
    FLAGS.num_steps_per_epoch = 400
    FLAGS.do_eval = False
    self._run_and_report_benchmark()

  # Short 1GPU throughput test for XLA benchmark
  def benchmark_short_1_gpu(self):
    super(BertClassifyBenchmarkReal, self)._setup()
    self.timer_callback = BenchmarkTimerCallback()
    FLAGS.max_seq_length = 512
    FLAGS.max_predictions_per_seq = 76
    FLAGS.steps_per_loop = 1
    FLAGS.input_files = "REDACTEDpart-*"
    FLAGS.num_train_epochs = 1
    FLAGS.num_steps_per_epoch = 10
    FLAGS.optimizer_type = "lamb"
    FLAGS.do_eval = False
    FLAGS.device_warmup = False
    FLAGS.distribution_strategy = "mirrored"
    FLAGS.dtype = "fp16"
    FLAGS.loss_scale = "dynamic"
    FLAGS.num_gpus = 1
    FLAGS.train_batch_size = 6
    FLAGS.eval_batch_size = 6
    # To improve kernel launching across multiple GPUs and improve step time
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    base_data_dir = "REDACTEDpublic/bert_mlperf_data/"
    FLAGS.bert_config_file = base_data_dir + "bert_config.json"
    FLAGS.model_dir = self._get_model_dir("benchmark_short_1_gpu")
    self._run_and_report_benchmark(force_gpu_memory_alloc=False)

  def _set_tpu_common(self, benchmark_name):
    FLAGS.distribution_strategy = "tpu"
    FLAGS.dtype = "bf16"
    FLAGS.train_batch_size = 448
    FLAGS.eval_batch_size = 768
    FLAGS.steps_per_loop = math.ceil(500000.0 / FLAGS.train_batch_size)
    FLAGS.steps_before_eval_start = math.ceil(3000000.0 /
                                              FLAGS.train_batch_size)
    FLAGS.steps_between_eval = FLAGS.steps_per_loop
    # In cloud, checkpoint restoring happens in TPU VM so data path needs to
    # be in GCS.
    base_data_dir = "REDACTEDpublic/bert_mlperf_data2/"
    FLAGS.bert_config_file = base_data_dir + "bert_config.json"
    FLAGS.model_dir = self._get_model_dir(benchmark_name)
    FLAGS.init_checkpoint = base_data_dir + "model.ckpt-28252"

  def benchmark_4x4_tpu(self):
    self._setup()
    self._set_tpu_common("benchmark_4x4_tpu")
    FLAGS.stop_steps = 8103
    FLAGS.num_steps_per_epoch = 8103
    FLAGS.epsilon = 1e-6
    FLAGS.beta_1 = 0.9
    FLAGS.beta_2 = 0.999
    FLAGS.learning_rate = 0.0004
    FLAGS.warmup_steps = 0
    FLAGS.weight_decay_rate = 0.01
    self._run_and_report_benchmark()

  def benchmark_short_4x4_tpu(self):
    self._setup()
    self._set_tpu_common("benchmark_short_4x4_tpu")
    FLAGS.init_checkpoint = None
    FLAGS.do_eval = False
    FLAGS.num_steps_per_epoch = FLAGS.steps_per_loop * 2
    self._run_and_report_benchmark()

if __name__ == "__main__":
  tf.test.main()
