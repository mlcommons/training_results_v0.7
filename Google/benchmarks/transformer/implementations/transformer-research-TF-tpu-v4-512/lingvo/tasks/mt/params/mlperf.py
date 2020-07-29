# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train NMT Models on WMT'14 English-German machine translation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import REDACTED.transformer_lingvo.lingvo.compat as tf
from REDACTED.transformer_lingvo.lingvo.core import ml_perf_tokenizer
from REDACTED.transformer_lingvo.lingvo import model_registry
from REDACTED.transformer_lingvo.lingvo.core import base_model_params
from REDACTED.transformer_lingvo.lingvo.core import layers
from REDACTED.transformer_lingvo.lingvo.core import program as program_lib
from REDACTED.transformer_lingvo.lingvo.core import py_utils
from REDACTED.transformer_lingvo.lingvo.core import schedule
from REDACTED.transformer_lingvo.lingvo.tasks.mt import base_config
from REDACTED.transformer_lingvo.lingvo.tasks.mt import input_generator
from REDACTED.transformer_lingvo.lingvo.tasks.mt import model

from REDACTED.transformer_lingvo.google import model_helper


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerf(base_model_params.SingleTaskModelParams):
  DATADIR = '/REDACTED/jk-d/home/tpu-perf-team/dehao/transformer_sharded_512/'

  ID_PAD = 0
  ID_SOS = 0
  ID_EOS = 1

  MODEL_DIM = 1024
  HIDDEN_DIM = 4096
  NUM_HEADS = 16
  VOCAB_SIZE = 33708
  WARMUP_STEPS = 1107
  LEARNING_RATE = 1.875
  MAX_STEPS = 10000

  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.98

  # Per-core batch_size
  TRAIN_BATCH_SIZE = 16
  DECODE_BATCH_SIZE = 96

  XLA_NUM_PARTITIONS = None
  NUM_HOSTS = None

  TEST_FILE_PATTERN = 'translate_ende_wmt32k-dev-3072-69-zero-eval-weight-?????-of-00512'

  def Train(self):
    p = input_generator.MlPerfInput.Params()
    p.tokenizer = ml_perf_tokenizer.MlPerfTokenizer.Params()
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.tokenizer.vocab_filepath = os.path.join(
        self.DATADIR, 'vocab.translate_ende_wmt32k.32768.subwords')
    p.tokenizer.target_eos_id = self.ID_EOS

    p.file_pattern = 'tfrecord:' + os.path.join(
        self.DATADIR, 'translate_ende_wmt32k-train-?????-of-00512')
    p = model_helper.FixateInputShape(p, self.TRAIN_BATCH_SIZE, 96, 96)

    p.num_samples = 0
    p.file_random_seed = 0
    p.use_per_host_infeed = True
    p.num_hosts = self.NUM_HOSTS
    return p

  def Test(self):
    p = input_generator.MlPerfInput.Params()
    p.tokenizer = ml_perf_tokenizer.MlPerfTokenizer.Params()
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.tokenizer.vocab_filepath = os.path.join(
        self.DATADIR, 'vocab.translate_ende_wmt32k.32768.subwords')
    p.tokenizer.target_eos_id = self.ID_EOS
    p.file_pattern = 'tfrecord:' + os.path.join(self.DATADIR,
                                                self.TEST_FILE_PATTERN)
    p = model_helper.FixateInputShape(p, self.DECODE_BATCH_SIZE, 97, 97)

    p.num_samples = 0
    # Note: Per host infeed for the test set evaluation does not seem
    # to be worth it, since the sharded files are so tiny, seems
    # the variance/overhead makes it a bad trade-off.
    p.use_per_host_infeed = True
    p.file_random_seed = 0
    p.num_hosts = self.NUM_HOSTS
    return p

  def _CommonParams(self, packed_input):
    p = base_config.SetupTransformerBatchMajorParams(
        name='en_de_ml_perf_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=self.MODEL_DIM,
        hidden_dim=self.HIDDEN_DIM,
        num_heads=self.NUM_HEADS,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='de',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        add_unnormalized_residuals=True,
        learning_rate=self.LEARNING_RATE,
        warmup_steps=self.WARMUP_STEPS,
        packed_input=packed_input,
        use_fast_projection_layer=True,
        enable_per_dim_scale=False,
        use_fused_layernorm=True,
        use_bf16_activations=py_utils.use_tpu(),
        use_bias=False,
        xla_num_partitions=self.XLA_NUM_PARTITIONS)

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    sm_params = p.decoder.softmax.Copy()
    sm_params.input_dim = self.MODEL_DIM
    shared_emb = layers.SharedSoftmaxLayer.Params().Set(
        **dict(sm_params.IterParams()))
    shared_emb.params_init = py_utils.WeightInit.Gaussian(
        1.0 / math.sqrt(self.MODEL_DIM))
    shared_emb.scale_sqrt_depth = True
    shared_emb.use_num_classes_major_weight = True
    p.decoder.shared_emb = shared_emb
    p.decoder.shared_emb.cls = layers.SharedSoftmaxLayer
    # Directly sharing encoder embedding with decoder results in worse model
    # quality, which requires more tuning.
    p.encoder.shared_emb = shared_emb
    p.encoder.shared_emb.cls = layers.SharedSoftmaxLayer

    p.train.lr_schedule = schedule.TransformerMLPerfSchedule.Params().Set(
        warmup_steps=self.WARMUP_STEPS, model_dim=self.MODEL_DIM)
    p.train.max_steps = self.MAX_STEPS
    p.train.scale_gradients = False

    p.train.optimizer.beta1 = self.ADAM_BETA1
    p.train.optimizer.beta2 = self.ADAM_BETA2

    # Fix this
    #p.eval.ml_perf_metrics_only = True

    p.decoder.beam_search.target_sos_id = self.ID_SOS
    p.decoder.beam_search.target_eos_id = self.ID_EOS

    p.decoder.beam_search.beam_size = 4.0
    p.decoder.beam_search.num_hyps_per_beam = 4

    p.decoder.target_sos_id = self.ID_SOS
    p.decoder.target_eos_id = self.ID_EOS

    p.decoder.use_fast_softmax = True

    p.decoder.target_seq_len = 147

    if py_utils.use_tpu():
      p.encoder.input_dropout_tpl.fprop_dtype = tf.bfloat16
      p.decoder.trans_decoder_tpl.fprop_dtype = tf.bfloat16
      p.decoder.input_dropout_tpl.fprop_dtype = tf.bfloat16
      p.train.optimizer.use_bf16_gradients_ar = True
    return p

  def Task(self):
    return self._CommonParams(packed_input=False)


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutor(WmtEnDeMlPerf):
  """TPU Executor params for MLPerf."""

  def Task(self):
    p = super(WmtEnDeMlPerfTpuExecutor, self).Task()
    p.decoder.beam_search = model_helper.ChangeToBeamSearchTpuHelper(
        p.decoder.beam_search)
    return p

  def ProgramSchedule(self):
    p = program_lib.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        # Sets train_steps_per_loop to the number of steps of an epoch based on
        # reference dataset size and 131k tokens global batch size (a typical
        # batch size from MLPerf 0.6 submission).
        train_steps_per_loop=1107,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=0,
        decode_steps_per_loop=1,
    )
    p.train_executions_per_eval = 3

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1107
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 96
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 580000
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorTuning(WmtEnDeMlPerfTpuExecutor):
  """TPU Executor tuning run for MLPerf.

  This run will take longer than a final run as we will
  pause training to decode more often, however
  it makes it easier to tune.

  """

  def ProgramSchedule(self):
    p = program_lib.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=200,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=0,
        decode_steps_per_loop=1,
    )
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1107
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 96
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 580000
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorPackedInputTuning(WmtEnDeMlPerfTpuExecutorTuning):
  """TPU Executor tuning run for MLPerf.
  """

  PACKED_DATADIR = '/REDACTED/pw-d/home/tpu-perf-team/blee/transformer_sharded_512/'
  PACKED_TRAIN_FILE_PATTERN = 'translate_ende_wmt32k_packed-train-?????-of-00256'

  def ProgramSchedule(self):
    p = program_lib.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=200,
        eval_dataset_names=['Test'],
        eval_steps_per_loop=0,
        decode_steps_per_loop=1,
    )
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1107
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p

  def Train(self):
    p = input_generator.MlPerfInput.Params()
    p.tokenizer = ml_perf_tokenizer.MlPerfTokenizer.Params()
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.tokenizer.vocab_filepath = os.path.join(
        self.PACKED_DATADIR, 'vocab.translate_ende_wmt32k.32768.subwords')
    p.tokenizer.target_eos_id = self.ID_EOS

    p.file_pattern = 'tfrecord:' + os.path.join(self.PACKED_DATADIR,
                                                self.PACKED_TRAIN_FILE_PATTERN)
    p = model_helper.FixateInputShape(p, self.TRAIN_BATCH_SIZE, 256, 256)

    p.num_samples = 0
    p.packed_input = True
    p.file_random_seed = 0
    p.use_per_host_infeed = True
    p.num_hosts = self.NUM_HOSTS
    return p

  def Task(self):
    p = self._CommonParams(packed_input=True)
    p.decoder.beam_search = model_helper.ChangeToBeamSearchTpuHelper(
        p.decoder.beam_search)
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShot(WmtEnDeMlPerfTpuExecutorPackedInputTuning
                                     ):
  """Run for e2e timing.

  This should hit .25 BLEU in ~3 epochs.
  """

  MAX_STEPS = 6 * 1107

  def ProgramSchedule(self):

    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1107,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=4)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1107
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShot16x16Partitioned(
    WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 16x16 DF."""

  MAX_STEPS = 6 * 1107
  TRAIN_BATCH_SIZE = 2
  DECODE_BATCH_SIZE = 12
  XLA_NUM_PARTITIONS = 2
  NUM_HOSTS = 32

  def Task(self):
    p = super(WmtEnDeMlPerfTpuExecutorOneShot16x16Partitioned, self).Task()
    p.xla_num_partitions = self.XLA_NUM_PARTITIONS
    return p

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1107,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=4)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1107
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShot16x16Tuning(WmtEnDeMlPerfTpuExecutorOneShot
                                                ):
  """Tuning runs."""

  WARMUP_STEPS = 831
  LEARNING_RATE = 1.875
  ADAM_BETA1 = 0.88
  ADAM_BETA2 = 0.92
  MAX_STEPS = 10 * 277
  TRAIN_BATCH_SIZE = 4
  DECODE_BATCH_SIZE = 6
  NUM_HOSTS = 32

  PACKED_DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_new_sharded_train80_data/'

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=277,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=7,
        warmup_seconds=0)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 277
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 2048
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = self.ADAM_BETA1
    p.ml_perf.opt_adam_beta_2 = self.ADAM_BETA2
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShot32x32Partitioned4(
    WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 32x32 DF."""

  WARMUP_STEPS = 550
  LEARNING_RATE = 1.3
  ADAM_BETA1 = 0.86
  ADAM_BETA2 = 0.92
  MAX_STEPS = 10 * 277
  TRAIN_BATCH_SIZE = 4
  DECODE_BATCH_SIZE = 6
  XLA_NUM_PARTITIONS = 4
  NUM_HOSTS = 128
  PACKED_DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_new_sharded_train80_data/'
  DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_test_data/'

  def Task(self):
    p = super(WmtEnDeMlPerfTpuExecutorOneShot32x32Partitioned4, self).Task()
    p.xla_num_partitions = self.XLA_NUM_PARTITIONS
    return p

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=277,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=7)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 277
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 2048
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = self.ADAM_BETA1
    p.ml_perf.opt_adam_beta_2 = self.ADAM_BETA2
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShotTwoPod(WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 64x32 DF."""

  WARMUP_STEPS = 831
  LEARNING_RATE = 1.875
  ADAM_BETA1 = 0.88
  ADAM_BETA2 = 0.92

  MAX_STEPS = 10 * 277
  TRAIN_BATCH_SIZE = 2
  DECODE_BATCH_SIZE = 3
  XLA_NUM_PARTITIONS = 4
  NUM_HOSTS = 256
  PACKED_DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_new_sharded_train80_data/'
  DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_test_data/'

  def Task(self):
    p = super(WmtEnDeMlPerfTpuExecutorOneShotTwoPod, self).Task()
    p.xla_num_partitions = self.XLA_NUM_PARTITIONS
    return p

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=277,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=7)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 277
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 2048
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = self.ADAM_BETA1
    p.ml_perf.opt_adam_beta_2 = self.ADAM_BETA2
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShotFourPod(WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 128x32 DF."""

  WARMUP_STEPS = 831
  LEARNING_RATE = 1.875
  ADAM_BETA1 = 0.88
  ADAM_BETA2 = 0.92
  TRAIN_BATCH_SIZE = 1
  DECODE_BATCH_SIZE = 2
  XLA_NUM_PARTITIONS = 4
  NUM_HOSTS = 512
  PACKED_DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_new_sharded_train80_data/'
  DATADIR = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/blee/transformer_test_data/'
  PACKED_TRAIN_FILE_PATTERN = 'translate_ende_wmt32k_packed-train-?????-of-00512'
  TEST_FILE_PATTERN = 'translate_ende_wmt32k-dev-4096-zero-eval-weight-00???-of-00512'

  def Task(self):
    p = super(WmtEnDeMlPerfTpuExecutorOneShotFourPod, self).Task()
    p.decoder.beam_search.short_seq_limit = 0
    p.xla_num_partitions = self.XLA_NUM_PARTITIONS
    return p

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=277,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=7,
        warmup_seconds=240)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 277
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 2048
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = self.ADAM_BETA1
    p.ml_perf.opt_adam_beta_2 = self.ADAM_BETA2
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShot16x16(WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 16x16 DF."""

  MAX_STEPS = 6 * 1107
  TRAIN_BATCH_SIZE = 1
  DECODE_BATCH_SIZE = 6
  NUM_HOSTS = 32

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1107,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=4)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1107
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShotPfc(WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 4x4x4 PFC."""

  STEPS_PER_EPOCH = 1107
  MAX_STEPS = 6 * 1107
  TRAIN_BATCH_SIZE = 4
  DECODE_BATCH_SIZE = 24
  NUM_HOSTS = 16
  PACKED_DATADIR = '/REDACTED/jk-d/home/tpu-perf-team/dehao/transformer_sharded_512/'
  PACKED_TRAIN_FILE_PATTERN = 'translate_ende_wmt32k_packed-train-?????-of-00016'
  TEST_FILE_PATTERN = 'translate_ende_wmt32k-dev-3072-69-zero-eval-weight-?????-of-00016'

  def ProgramSchedule(self):

    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1107,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=4)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = self.STEPS_PER_EPOCH
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShotPfcEightCube(
    WmtEnDeMlPerfTpuExecutorOneShot):
  """Run for 2x2x2 PFC."""

  STEPS_PER_EPOCH = 277
  WARMUP_STEPS = 831
  LEARNING_RATE = 1.9
  ADAM_BETA1 = 0.88
  ADAM_BETA2 = 0.94
  MAX_STEPS = 10 * 277
  TRAIN_BATCH_SIZE = 4
  DECODE_BATCH_SIZE = 2
  NUM_HOSTS = 64
  PACKED_DATADIR = '/REDACTED/jk-d/home/tpu-perf-team/dehao/transformer_sharded_512/'
  PACKED_TRAIN_FILE_PATTERN = 'translate_ende_wmt32k_packed-train-?????-of-00512'
  TEST_FILE_PATTERN = 'translate_ende_wmt32k-dev-4096-zero-eval-weight-00???-of-00512'

  def ProgramSchedule(self):

    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=self.STEPS_PER_EPOCH,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=7)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = self.STEPS_PER_EPOCH
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 2048
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = self.ADAM_BETA1
    p.ml_perf.opt_adam_beta_2 = self.ADAM_BETA2
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMlPerfTpuExecutorOneShotPfcTwoCube(WmtEnDeMlPerfTpuExecutorOneShot
                                               ):
  """Run for 2x2x2 PFC."""

  STEPS_PER_EPOCH = 1107
  MAX_STEPS = 6 * 1107
  TRAIN_BATCH_SIZE = 32
  DECODE_BATCH_SIZE = 24 * 8
  NUM_HOSTS = 2
  PACKED_DATADIR = '/REDACTED/jk-d/home/tpu-perf-team/dehao/transformer_sharded_512/'
  PACKED_TRAIN_FILE_PATTERN = 'translate_ende_wmt32k_packed-train-?????-of-00016'
  TEST_FILE_PATTERN = 'translate_ende_wmt32k-dev-3072-69-zero-eval-weight-?????-of-00016'

  def ProgramSchedule(self):

    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=1107,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=4)
    p.train_executions_per_eval = 1

    # For compliance logging.
    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = self.STEPS_PER_EPOCH
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    p.ml_perf.decoder_metric_success_threshold = 0.25

    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 512
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = self.LEARNING_RATE
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnDeMLPerfTinyTPUExecutor(WmtEnDeMlPerfTpuExecutor):
  """Tiny Transformer for TPUExecutor test."""

  MODEL_DIM = 4
  HIDDEN_DIM = 4
  NUM_HEADS = 2

  def ProgramSchedule(self):
    p = program_lib.MLPerfProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=20,
        decode_dataset_name='Test',
        decode_steps_per_loop=1,
        num_epochs_per_session_run=4,
        warmup_seconds=0)

    p.train_executions_per_eval = 1

    p.ml_perf.benchmark_name = 'transformer'
    p.ml_perf.steps_per_epoch = 1
    p.ml_perf.decoder_metric_name = 'ml_perf_bleu'
    # Dummy value just to see run_stop/success.
    p.ml_perf.decoder_metric_success_threshold = -10.0
    p.ml_perf.max_sequence_length = 80
    p.ml_perf.global_batch_size = 64
    p.ml_perf.optimizer_name = 'adam'
    p.ml_perf.opt_adam_beta_1 = 0.9
    p.ml_perf.opt_adam_beta_2 = 0.98
    p.ml_perf.opt_adam_epsilon = 1e-9
    p.ml_perf.base_learning_rate = 2.0
    p.ml_perf.warmup_steps = self.WARMUP_STEPS
    p.ml_perf.train_samples = 566340
    p.ml_perf.eval_samples = 3003
    return p
