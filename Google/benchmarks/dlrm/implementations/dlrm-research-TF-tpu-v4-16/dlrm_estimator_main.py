# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DLRM implementation with REDACTED embeddings via TPUEstimator."""

import os
import timeit

import REDACTED

from absl import app as absl_app
from absl import flags
import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.contrib.tpu.python.tpu import async_checkpoint

from REDACTED.tensorflow_models.mlperf.models.rough.dlrm import dataloader
from REDACTED.tensorflow_models.mlperf.models.rough.dlrm import dlrm
from REDACTED.tensorflow_models.mlperf.models.rough.dlrm import feature_config as fc
from REDACTED.tensorflow_models.mlperf.models.rough.dlrm_tf2 import common

FLAGS = flags.FLAGS

flags.DEFINE_string("master", default=None, help="Address of the master.")
flags.DEFINE_string(name="model_dir", default=None, help="Model directory.")


def create_tpu_estimator_columns(feature_columns, params, iters_per_loop=200):
  """Creates TPU estimator using feature columns.

  Args:
    feature_columns: Feature columns to use.
    params: Hparams for the model.
    iters_per_loop: Number of iterations to use per device loop invocation.

  Returns:
    An instance of TPUEstimator to use when training model.
  """
  dlrm_tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=iters_per_loop,
      per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
      .PER_HOST_V2)
  run_config = tf.estimator.tpu.RunConfig(
      master=FLAGS.master, tpu_config=dlrm_tpu_config)
  embedding_config_spec = tf.estimator.tpu.experimental.EmbeddingConfigSpec(
      feature_columns=feature_columns,
      optimization_parameters=tf.tpu.experimental
      .StochasticGradientDescentParameters(learning_rate=FLAGS.learning_rate),
      pipeline_execution_with_tensor_core=FLAGS.pipeline_execution,
      partition_strategy=FLAGS.partition_strategy)
  # Key "batch_size" is reserved by TPUEstimator.
  tpu_params = {k: v for k, v in params.items() if k != "batch_size"}
  return tf.estimator.tpu.TPUEstimator(
      model_fn=dlrm.create_model_fn(),
      config=run_config,
      use_tpu=True,
      train_batch_size=params["batch_size"],
      params=tpu_params,
      model_dir=FLAGS.model_dir,
      embedding_config_spec=embedding_config_spec)


def create_tpu_estimator_dicts(feature_to_config_dict,
                               table_to_config_dict,
                               params,
                               iters_per_loop=200):
  """Creates TPU estimator using feature config dicts.

  Args:
    feature_to_config_dict: Feature config dicts using TableConfig values.
    table_to_config_dict: Feature config dicts using FeatureConfig values.
    params: Hparams for the model.
    iters_per_loop: Number of iterations to use per device loop invocation.

  Returns:
    An instance of TPUEstimator to use when training model.
  """
  per_host_train = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  # SLICED: hangs - not supported with REDACTED?
  # PER_HOST_V1: the batch dimension of the dense inputs is not sharded

  # per_host_eval = tf.estimator.tpu.InputPipelineConfig.SLICED
  per_host_eval = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V1
  # per_host_eval = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2

  dlrm_tpu_config = tf.estimator.tpu.TPUConfig(
      iterations_per_loop=iters_per_loop,
      per_host_input_for_training=per_host_train,
      eval_training_input_configuration=per_host_eval,
      experimental_host_call_every_n_steps=FLAGS.summary_every_n_steps)

  run_config = tf.estimator.tpu.RunConfig(
      master=FLAGS.master,
      model_dir=FLAGS.model_dir,
      # Disable checkpointing and use async checkpointing instead.
      save_checkpoints_steps=None,
      save_checkpoints_secs=None,
      log_step_count_steps=FLAGS.summary_every_n_steps,
      tpu_config=dlrm_tpu_config
  )

  embedding_config_spec = tf.estimator.tpu.experimental.EmbeddingConfigSpec(
      table_to_config_dict=table_to_config_dict,
      feature_to_config_dict=feature_to_config_dict,
      optimization_parameters=tf.tpu.experimental
      .StochasticGradientDescentParameters(learning_rate=FLAGS.learning_rate),
      pipeline_execution_with_tensor_core=FLAGS.pipeline_execution,
      # (for quality) gradient_multiplier
      partition_strategy=FLAGS.partition_strategy,
  )
  # Key "batch_size" is reserved by TPUEstimator.
  tpu_params = {k: v for k, v in params.items() if k != "batch_size"}
  return tf.estimator.tpu.TPUEstimator(
      model_fn=dlrm.create_model_fn(),
      config=run_config,
      use_tpu=True,
      train_batch_size=params["batch_size"],
      eval_batch_size=params["eval_batch_size"],
      params=tpu_params,
      embedding_config_spec=embedding_config_spec)


def load_global_step_from_checkpoint_dir(checkpoint_dir):
  try:
    checkpoint_reader = tf.training.NewCheckpointReader(
        tf.training.latest_checkpoint(checkpoint_dir))
    return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
  except:  # pylint: disable=bare-except
    return 0


def main(_):
  params = common.get_params()
  feature_to_config_dict, table_to_config_dict = fc.get_feature_tbl_config(
      params)

  # Builds an estimator using FeatureConfig and TableConfig, as defined in
  # third_party/tensorflow/python/tpu/tpu_embedding.py
  estimator = create_tpu_estimator_dicts(
      feature_to_config_dict,
      table_to_config_dict,
      params,
      iters_per_loop=FLAGS.summary_every_n_steps)

  train_input_fn = dataloader.CriteoTsvReader(
      file_path="/REDACTED/mb-d/home/tpu-perf-team/tayo/criteo/terabyte_mlperf/rs=6.3/train/terabyte_train*",
      is_training=True,
      use_synthetic_data=params["use_synthetic_data"])
  eval_input_fn = dataloader.CriteoTsvReader(
      file_path="/readahead/128M/REDACTED/iz-d/home/tpu-perf-team/tayo/criteo/terabyte_mlperf/rs=6.3/eval/terabyte_eval*",
      is_training=False,
      use_synthetic_data=params["use_synthetic_data"])

  if FLAGS.mode == "eval":
    # From Pytorch logging:
    # num eval batches: 1361, each 64K
    # num train batches: 64014, each 64K
    # 64013*4 @ 16k
    # From other source:
    # num_train_samples = 4195197692

    if params["terabyte"]:
      # TODO(tayo): The following number drops remainder.
      num_eval_records = 89128960
      num_eval_steps = num_eval_records // FLAGS.eval_batch_size

    cycle_idx = 0

    # Run evaluation when there appears a new checkpoint.
    for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, timeout=None):
      try:
        tf.logging.info("Beginning eval iteration {}.".format(cycle_idx + 1))
        cycle_idx = cycle_idx + 1

        start_time = timeit.default_timer()
        eval_metrics = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=num_eval_steps,
            checkpoint_path=ckpt
            # checkpoint_path="/REDACTED/mb-d/home/tpu-perf-team/tayo/dlrm/model_dir_full_precision_0/model.ckpt-256000",
        )
        tf.logging.info(
            "Eval results: {}. Elapsed eval time: {:.4f}".format(
                eval_metrics,
                timeit.default_timer() - start_time))

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split("-")[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              "Evaluation finished after training step %d", current_step)
          break
      except tf.errors.NotFoundError:
        tf.logging.info(
            "Checkpoint %s no longer exists, skipping checkpoint", ckpt)
  else:   # FLAGS.mode == "train"
    current_step = load_global_step_from_checkpoint_dir(FLAGS.model_dir)

    tf.logging.info("Training for {} steps at batch_size {}.".format(
        FLAGS.train_steps, FLAGS.batch_size))
    start_time = timeit.default_timer()
    hooks = []
    hooks.append(
        async_checkpoint.AsyncCheckpointSaverHook(
            checkpoint_dir=FLAGS.model_dir,
            save_steps=128000))
    estimator.train(
        input_fn=train_input_fn,
        max_steps=FLAGS.train_steps,
        hooks=hooks
    )

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  common.define_dlrm_flags()
  absl_app.run(main)
