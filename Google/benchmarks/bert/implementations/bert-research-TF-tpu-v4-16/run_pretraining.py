"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from REDACTED.tensorflow_models.mlperf.models.rough.bert import dataset_input
from REDACTED.tensorflow_models.mlperf.models.rough.bert import modeling
from REDACTED.tensorflow_models.mlperf.models.rough.bert import optimization
from REDACTED.tensorflow_models.mlperf.models.rough.mlp_log import mlp_log
from REDACTED.tensorflow_models.mlperf.models.rough.util import train_and_eval_runner

from REDACTED.learning.REDACTED.google.python.training import hparam

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string("local_prefix", None,
                    "Path to the local directory that stores training data.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "adamw", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate.")

flags.DEFINE_float(
    "lamb_weight_decay_rate", default=0.01, help=("Hyperparameter for LAMB."))

flags.DEFINE_float(
    "lamb_beta_1", default=0.9, help=("Hyperparameter for LAMB."))

flags.DEFINE_float(
    "lamb_beta_2", default=0.999, help=("Hyperparameter for LAMB."))

flags.DEFINE_integer(
    "log_epsilon", default=-6, help=("Hyperparameter for Optimizers."))

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_integer("num_train_steps", 100000,
                     "Number of training steps, used to scale the LR.")

flags.DEFINE_integer("stop_steps", 10000, "Number of training steps to stop.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_integer("num_eval_samples", 10000, "Number of eval samples.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("use_bfloat16_activation", True, "Whether to use bfloat16 "
                  "for activations on TPU.")

flags.DEFINE_bool("use_bfloat16_all_reduce", False, "Whether to use bfloat16 "
                  "for all reduce on TPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float(
    "stop_threshold", default=0.714, help=("Stop threshold for MLPerf."))

flags.DEFINE_integer(
    "num_gpus", 0,
    "Use the GPU backend if this value is set to more than zero.")

flags.DEFINE_bool("repeatable", False, "Fix the random seed to avoid run-to-run"
                  "variations when measuring computing performance.")

flags.DEFINE_integer(
    "num_partitions", default=1, help=("Number of SPMD Partitions."))

flags.DEFINE_bool(
    "best_effort_init", default=False,
    help=("Init only the variables that exist in the checkpoint."
          "This can be turn on for model changes that contain "
          "extra variables than original checkpoint."))

flags.DEFINE_integer("steps_per_update", 1,
                     "The number of steps for accumulating gradients.")


def bert_model_fn(features, labels, is_training):  # pylint: disable=unused-argument
  """The `model_fn` for LowLevelRunner."""

  tf.logging.info("*** Features ***")
  for name in sorted(features.keys()):
    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

  input_ids = features["input_ids"]
  input_mask = features["input_mask"]
  segment_ids = features["segment_ids"]
  masked_lm_positions = features["masked_lm_positions"]
  masked_lm_ids = features["masked_lm_ids"]
  masked_lm_weights = features["masked_lm_weights"]
  next_sentence_labels = features["next_sentence_labels"]

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  use_one_hot_embeddings = False
  learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  num_warmup_steps = FLAGS.num_warmup_steps
  start_warmup_step = FLAGS.start_warmup_step
  num_train_steps = FLAGS.num_train_steps
  use_tpu = FLAGS.use_tpu
  optimizer = FLAGS.optimizer
  poly_power = FLAGS.poly_power
  lamb_weight_decay_rate = FLAGS.lamb_weight_decay_rate
  lamb_beta_1 = FLAGS.lamb_beta_1
  lamb_beta_2 = FLAGS.lamb_beta_2
  log_epsilon = FLAGS.log_epsilon

  tf.logging.info("Using learning rate: %s", learning_rate)
  print("Using learning rate:", learning_rate)
  tf.logging.info("Using lamb_weight_decay_rate: %s", lamb_weight_decay_rate)
  print("Using lamb_weight_decay_rate:", lamb_weight_decay_rate)
  tf.logging.info("Using beta 1: %s", lamb_beta_1)
  print("Using beta 1:", lamb_beta_1)
  tf.logging.info("Using beta 2: %s", lamb_beta_2)
  print("Using beta 2:", lamb_beta_2)
  tf.logging.info("Using log_epsilon: %s", log_epsilon)
  print("Using log_epsilon:", log_epsilon)
  tf.logging.info("Using num_warmup_steps: %s", num_warmup_steps)
  print("Using num_warmup_steps:", num_warmup_steps)
  tf.logging.info("Using num_train_steps: %s", num_train_steps)
  print("Using num_train_steps:", num_train_steps)

  tf.get_variable_scope().set_custom_getter(
      modeling.bfloat16_var_getter if FLAGS.use_bfloat16_activation else None)

  if FLAGS.use_bfloat16_activation:
    tf.logging.info("Using bfloat16 for activations.")

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_bfloat16_activation=FLAGS.use_bfloat16_activation,
      num_partitions=FLAGS.num_partitions)

  (masked_lm_loss, masked_lm_example_loss,
   masked_lm_log_probs) = get_masked_lm_output(
       bert_config, tf.cast(model.get_sequence_output(), tf.float32),
       model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
       masked_lm_weights, FLAGS.num_partitions)

  (
      next_sentence_loss,
      _,  #  next_sentence_example_loss,
      _  #  next_sentence_log_probs
  ) = get_next_sentence_output(bert_config,
                               tf.cast(model.get_pooled_output(), tf.float32),
                               next_sentence_labels)

  total_loss = masked_lm_loss + next_sentence_loss

  if not is_training:
    # Computes the loss and accuracy of the model.
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])

    masked_lm_weighted_correct = tf.multiply(
        tf.cast(tf.equal(masked_lm_ids, masked_lm_predictions), tf.float32),
        masked_lm_weights)
    masked_lm_weighted_correct = tf.reduce_sum(masked_lm_weighted_correct)
    masked_lm_weighted_count = tf.reduce_sum(masked_lm_weights)

    return None, {
        "masked_lm_weighted_correct":
            tf.reshape(masked_lm_weighted_correct, [-1]),
        "masked_lm_weighted_count":
            tf.reshape(masked_lm_weighted_count, [-1])}

  train_op = optimization.create_optimizer(
      total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
      optimizer, poly_power, start_warmup_step, lamb_weight_decay_rate,
      lamb_beta_1, lamb_beta_2, log_epsilon, FLAGS.use_bfloat16_all_reduce,
      FLAGS.steps_per_update)

  return train_op, None


def get_masked_lm_output(bert_config,
                         input_tensor,
                         output_weights,
                         positions,
                         label_ids,
                         label_weights,
                         num_partitions=1):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions", reuse=tf.AUTO_REUSE):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    if num_partitions > 1:
      output_weights = xla_sharding.split(
          output_weights, 0, num_partitions, use_sharding_op=True)
      output_bias = xla_sharding.split(
          output_bias, 0, num_partitions, use_sharding_op=True)
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship", reuse=tf.AUTO_REUSE):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_scaffold_fn(init_checkpoint):
  """Get scaffold function."""
  tvars = tf.trainable_variables()

  initialized_variable_names = {}

  scaffold_fn = None
  if init_checkpoint:
    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    def tpu_scaffold():
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      return tf.train.Scaffold()
    for var in initialized_variable_names:
      tf.logging.info("Inited: %s", var)
    for k in assignment_map:
      tf.logging.info("Assigment: %s => %s", k, assignment_map[k])

    scaffold_fn = tpu_scaffold

  tf.logging.info("**** Trainable Variables ****")
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)

  return scaffold_fn


masked_lm_accuracy = 0
run_steps = 0


def run_pretraining(hparams):
  """Run pretraining with given hyperparameters."""

  global masked_lm_accuracy
  global run_steps

  masked_lm_accuracy = 0
  run_steps = 0

  def eval_init_fn(cur_step):
    """Executed before every eval."""
    # While BERT pretraining does not have epochs,
    # to make the logging consistent with other mlperf models,
    # in all the mlp_log, epochs are steps, and examples are sequences.
    mlp_log.mlperf_print(
        "block_start",
        None,
        metadata={
            "first_epoch_num": cur_step + FLAGS.iterations_per_loop,
            "epoch_count": FLAGS.iterations_per_loop
        })

  def eval_finish_fn(cur_step, eval_output, summary_writer):
    """Executed after every eval."""
    global run_steps
    global masked_lm_accuracy
    cur_step_corrected = cur_step + FLAGS.iterations_per_loop
    run_steps = cur_step_corrected
    masked_lm_weighted_correct = eval_output["masked_lm_weighted_correct"]
    masked_lm_weighted_count = eval_output["masked_lm_weighted_count"]

    masked_lm_accuracy = np.sum(masked_lm_weighted_correct) / np.sum(
        masked_lm_weighted_count)
    # the eval_output may mix up the order of the two arrays
    # swap the order if it did got mix up
    if masked_lm_accuracy > 1:
      masked_lm_accuracy = 1 / masked_lm_accuracy

    if summary_writer:
      with tf.Graph().as_default():
        summary_writer.add_summary(
            tf.Summary(value=[
                tf.Summary.Value(tag="masked_lm_accuracy",
                                 simple_value=masked_lm_accuracy)
            ]), cur_step_corrected)

    mlp_log.mlperf_print(
        "block_stop",
        None,
        metadata={
            "first_epoch_num": cur_step_corrected,
        })
    # While BERT pretraining does not have epochs,
    # to make the logging consistent with other mlperf models,
    # in all the mlp_log, epochs are steps, and examples are sequences.
    mlp_log.mlperf_print(
        "eval_accuracy",
        float(masked_lm_accuracy),
        metadata={"epoch_num": cur_step_corrected})
    if (masked_lm_accuracy >= FLAGS.stop_threshold and
        cur_step_corrected >= FLAGS.iterations_per_loop * 6):
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
      return True
    else:
      return False

  def run_finish_fn(success):
    if not success:
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "abort"})
    mlp_log.mlperf_print("run_final", None)

  def init_fn():
    if FLAGS.init_checkpoint:
      if FLAGS.best_effort_init:
        variable_pair_list = tf.train.list_variables(FLAGS.init_checkpoint)
        variables = {name for name, _ in variable_pair_list}
        assignment_map = {}
        for v in tf.global_variables():
          name = v.op.name
          if name in variables and name.split("/", 1)[0] in ("bert", "cls"):
            assignment_map[name] = v
        tf.logging.info("len(assignment_map) = %d", len(assignment_map))
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
      else:
        tf.train.init_from_checkpoint(FLAGS.init_checkpoint, {
            "bert/": "bert/",
            "cls/": "cls/",
        })

  # Passing the hyperparameters
  if "learning_rate" in hparams:
    FLAGS.learning_rate = hparams.learning_rate
  if "lamb_weight_decay_rate" in hparams:
    FLAGS.lamb_weight_decay_rate = hparams.lamb_weight_decay_rate
  if "lamb_beta_1" in hparams:
    FLAGS.lamb_beta_1 = hparams.lamb_beta_1
  if "lamb_beta_2" in hparams:
    FLAGS.lamb_beta_2 = hparams.lamb_beta_2
  if "epsilon" in hparams:
    FLAGS.epsilon = hparams.epsilon
  if "num_warmup_steps" in hparams:
    FLAGS.num_warmup_steps = hparams.num_warmup_steps
  if "num_train_steps" in hparams:
    FLAGS.num_train_steps = hparams.num_train_steps

  # Input handling
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.repeatable:
    tf.set_random_seed(123)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    if FLAGS.local_prefix:
      for f in tf.gfile.Glob(input_pattern):
        input_files.append(
            os.path.join(FLAGS.local_prefix, os.path.basename(f)))
    else:
      input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  tf.logging.info("%s Files." % len(input_files))

  dataset_train = dataset_input.input_fn_builder(
      input_files=input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=True,
      num_cpu_threads=8)

  dataset_eval = dataset_input.input_fn_builder(
      input_files=input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=False,
      num_cpu_threads=8,
      num_eval_samples=FLAGS.num_eval_samples)

  # Create the low level runner
  low_level_runner = train_and_eval_runner.TrainAndEvalRunner(
      FLAGS.iterations_per_loop, FLAGS.stop_steps + 1, FLAGS.max_eval_steps,
      FLAGS.num_tpu_cores // FLAGS.num_partitions)

  mlp_log.mlperf_print("cache_clear", True)
  mlp_log.mlperf_print("init_start", None)
  mlp_log.mlperf_print("global_batch_size", FLAGS.train_batch_size)
  mlp_log.mlperf_print("opt_learning_rate_warmup_steps", FLAGS.num_warmup_steps)
  mlp_log.mlperf_print("num_warmup_steps", FLAGS.num_warmup_steps)
  mlp_log.mlperf_print("start_warmup_step", FLAGS.start_warmup_step)
  mlp_log.mlperf_print("max_sequence_length", FLAGS.max_seq_length)
  mlp_log.mlperf_print("opt_base_learning_rate", FLAGS.learning_rate)
  mlp_log.mlperf_print("opt_lamb_beta_1", FLAGS.lamb_beta_1)
  mlp_log.mlperf_print("opt_lamb_beta_2", FLAGS.lamb_beta_2)
  mlp_log.mlperf_print("opt_epsilon", 10 ** FLAGS.log_epsilon)
  mlp_log.mlperf_print("opt_learning_rate_training_steps",
                       FLAGS.num_train_steps)
  mlp_log.mlperf_print("opt_lamb_weight_decay_rate",
                       FLAGS.lamb_weight_decay_rate)
  mlp_log.mlperf_print("opt_lamb_learning_rate_decay_poly_power", 1)
  mlp_log.mlperf_print("opt_gradient_accumulation_steps", 0)
  mlp_log.mlperf_print("max_predictions_per_seq", FLAGS.max_predictions_per_seq)

  low_level_runner.initialize(
      dataset_train,
      dataset_eval,
      bert_model_fn,
      FLAGS.train_batch_size,
      FLAGS.eval_batch_size,
      input_partition_dims=None,
      init_fn=init_fn,
      train_has_labels=False,
      eval_has_labels=False,
      num_partitions=FLAGS.num_partitions)

  mlp_log.mlperf_print("init_stop", None)

  mlp_log.mlperf_print("run_start", None)

  # To make the logging consistent with other mlperf models,
  # in all the mlp_log, epochs are steps, and examples are sequences.
  mlp_log.mlperf_print("train_samples",
                       FLAGS.num_train_steps * FLAGS.train_batch_size)
  mlp_log.mlperf_print("eval_samples", FLAGS.num_eval_samples)
  low_level_runner.train_and_eval(eval_init_fn, eval_finish_fn, run_finish_fn)
  return masked_lm_accuracy, run_steps


def main(unused_argv):
  # Input handling
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.repeatable:
    tf.set_random_seed(123)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  default_hparams = hparam.HParams(learning_rate=FLAGS.learning_rate)

  accuracy, steps = run_pretraining(default_hparams)
  print("Returned accuracy:", accuracy)
  print("Returned steps:", steps)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("bert_config_file")
  app.run(main)
