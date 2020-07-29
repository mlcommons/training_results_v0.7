# Lint as: python3
"""Script to train BERT model on pretrain task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent.futures import thread
import functools
import os
import time
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
import jax

from jax import config
from jax import lax
from jax import random
import jax.numpy as jnp
from jax.util import partial

import numpy as np
import tensorflow.compat.v2 as tf

import REDACTED.learning.deepmind.xmanager2.client.google as xm
from REDACTED import xprof_session
from REDACTED import gfile

from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import bert_lamb
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import bert_models
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import input_pipeline
from REDACTED.tensorflow_models.mlperf.models.rough.bert_jax import utils
from REDACTED.tensorflow_models.mlperf.models.rough.mlp_log import mlp_log

CONFIG_FILE = '/REDACTED/od-d/home/jacobdevlin/public/bert/pretrained_models/uncased_L-24_H-1024_A-16/bert_config.json'
CHECKPOINT = '/REDACTED/je-d/home/staging-REDACTED-gpu-dedicated/bert/pretrained_model/converted'
INPUT_FILES = 'REDACTEDpart-*'

flags.DEFINE_float('target_accuracy', 0.712, help='Base learning rate.')

flags.DEFINE_integer(
    'checkpoint_freq', default=10000,
    help='Whether to restore from existing model checkpoints.')

flags.DEFINE_string(
    'model_dir',
    default=None,
    help=('The directory where the model and summaries are stored.'))
flags.DEFINE_float('learning_rate', 1e-4, help='Base learning rate.')

flags.DEFINE_string('eval_input_files', INPUT_FILES,
                    'File path to retrieve eval data for pre-training.')

flags.DEFINE_string('input_files', INPUT_FILES,
                    'File path to retrieve training data for pre-training.')
flags.DEFINE_string('bert_config_file', CONFIG_FILE,
                    'Config file for BERT model.')
flags.DEFINE_string('init_checkpoint', CHECKPOINT,
                    'Checkpoint of pretained model.')
flags.DEFINE_integer('train_batch_size', 4, 'Global batch size for training.')
flags.DEFINE_integer('eval_batch_size', 8, 'Global batch size for eval.')
flags.DEFINE_integer('max_eval_steps', 8, 'Steps for eval.')
flags.DEFINE_integer('num_steps_per_epoch', 250,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_bool('save_checkpoint', False,
                  'Whether to save current training model checkpoints.')
flags.DEFINE_bool('load_checkpoint', False,
                  'Whether to restore from existing model checkpoints.')

flags.DEFINE_boolean('load_tf_weights', True,
                     'Load tensorflow pretrained weights into JAX model.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for Adam/LAMB weight decay optimizer.')
flags.DEFINE_bool('use_lamb', True, 'Whether use LAMB optimizer')
flags.DEFINE_bool('load_mlperf_weights', True,
                  'Whether to load mlperf weights or keras BERT weights.')
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_float('poly_power', 1.0, 'The power of poly decay.')
flags.DEFINE_integer('start_warmup_step', 0, 'The starting step of warmup.')
flags.DEFINE_integer('total_training_steps', 100000,
                     'Number of training steps.')
flags.DEFINE_bool('precompile', True, 'Compile beforehand with fake input')
flags.DEFINE_bool('profile', default=True,
                  help='Enable programmatic profile with xprof.')
flags.DEFINE_bool('end_to_end_profile', default=False,
                  help='Enable programmatic end_to_end profile with xprof.')

flags.DEFINE_float('profile_duration', default=0.2,
                   help='How long in seconds to profile')
flags.DEFINE_integer('profile_latency', default=30,
                     help='How long after the training loop to start profile')
flags.DEFINE_bool(
    'hardware_rng', default=True,
    help='Whether to use hardware rng for dropout.')
flags.DEFINE_bool(
    'infeed',
    default=True,
    help='Stage out training loop to XLA using infeed for data loading.')

flags.DEFINE_integer('eval_sample_size', default=10000,
                     help='Global number of samples for eval dataset')
flags.DEFINE_bool('use_bfloat16_activation', True, 'Whether to use bfloat16 '
                  'for activations on TPU.')
flags.DEFINE_bool('reduce_gradients_in_bf16', True, 'Whether to use bfloat16 '
                  'for gradient all-reduce.')

flags.DEFINE_bool('enable_wus', True, 'Enable Weight update sharding')
flags.DEFINE_bool('enable_buffer_donation', False,
                  'Enable buffer donation.')
flags.DEFINE_integer('repeat_experiment', 1,
                     'How many times to repeat the experiment.')

flags.DEFINE_integer('init_sleep', 0,
                     'Sleep seconds between compilation and run start.')
flags.DEFINE_integer('seed', None, 'Random seed')

flags.DEFINE_float(
    'lamb_beta_1', default=0.9, help=('Hyperparameter for LAMB.'))

flags.DEFINE_float(
    'lamb_beta_2', default=0.999, help=('Hyperparameter for LAMB.'))

flags.DEFINE_integer(
    'log_epsilon', default=-6, help=('Hyperparameter for Optimizers.'))

flags.DEFINE_float(
    'lamb_weight_decay', default=0.01, help=('Weight decay for LAMB.'))

# Adds jax_log_compiles flag to print compilation logs on the jax side.
config.parse_flags_with_absl()
FLAGS = flags.FLAGS
RUN_STOP = False
TOTAL_STEPS = False


def _unbroadcast(x):
  """Assuming `x` is replicated along its leading axis, remove that axis."""
  assert isinstance(x, jax.pxla.ShardedDeviceArray)
  sharding_spec = x.sharding_spec
  assert sharding_spec.shards_per_axis[0] == x.shape[0]
  assert not sharding_spec.is_axis_materialized[0]
  aval = jax.abstract_arrays.ShapedArray(x.shape[1:], x.dtype)
  sharding_spec = jax.pxla.ShardingSpec(
      sharding_spec.shards_per_axis[1:],
      sharding_spec.is_axis_materialized[1:],
      [(x.shape[0], 0)] + [(factor, index - 1) for factor, index
                           in sharding_spec.replication_factors])
  return jax.pxla.ShardedDeviceArray(aval, sharding_spec, x.device_buffers)


def unbroadcast(tree):
  """Assuming `tree` is replicated along its leading axis, remove that axis."""
  return jax.tree_map(_unbroadcast, tree)


def hardware_bernoulli(rng_key, p=np.float32(0.5), shape=None):
  """Faster RNG."""
  y = 1.0
  x = 0.0
  if FLAGS.use_bfloat16_activation:
    y = jnp.bfloat16(y)
    x = jnp.bfloat16(0.0)
    p = jnp.bfloat16(p)
  y = lax.tie_in(rng_key, y)
  m = lax.rng_uniform(x, y, shape)
  if FLAGS.use_bfloat16_activation:
    assert m.dtype == jnp.bfloat16
  return m < p


def set_hardware_bernoulli():
  jax.random.bernoulli = hardware_bernoulli


def xprof_profile(start_after_sec=30, profile_time_sec=1,
                  device='REDACTED'):
  """Profiles single host with after start_after_sec for profile_time_sec.

  Args:
    start_after_sec: when to start profiling in sec.
    profile_time_sec: how long to profile in sec.
    device: string, one of ['', 'cpu', 'gpu', 'REDACTED', 'REDACTED'].
  """
  if device not in  ['', 'cpu', 'gpu', 'REDACTED', 'REDACTED']:
    logging.error('Incorrect device for profiling %s', device)
    return
  time.sleep(start_after_sec)
  xprof = xprof_session.XprofSession()
  xprof.start_session(device_name=device,
                      enable_python_tracer=True,
                      host_trace_level=2)
  time.sleep(profile_time_sec)
  xprof_url = xprof.end_session_and_get_url(tag='')
  logging.info('Xprof profile is at %s', xprof_url)


def profile_with_xprof_on_background(start_after_sec=30, profile_time_sec=1,
                                     device='REDACTED'):
  profiler_thread = thread.ThreadPoolExecutor(jax.local_device_count(), 'xprof')
  profiler_thread.submit(partial(xprof_profile, start_after_sec,
                                 profile_time_sec, device))


def compute_weighted_cross_entropy(logits,
                                   labels):
  """Compute weighted cross entropy and entropy for log probs and labels.

  Args:
   logits: [batch, length, num_classes] float array.
   labels: categorical targets [batch, length] int array.
  Returns:
    Tuple of scalars of loss and per example loss.
  """
  log_probs = nn.log_softmax(logits)
  labels = jnp.reshape(labels, [-1])
  one_hot_labels = common_utils.onehot(labels, num_classes=2)
  per_example_loss = -jnp.sum(one_hot_labels * log_probs, axis=-1)
  loss = jnp.mean(per_example_loss)
  return (loss, per_example_loss)


def get_masked_lm_output(logits, label_ids, label_weights):
  """Calculate masked_lm loss for pretrain task."""
  vocab_size = logits.shape[-1]

  label_ids = jnp.reshape(label_ids, (-1))
  label_weights = jnp.reshape(label_weights, (-1))
  one_hot_labels = common_utils.onehot(
      label_ids, vocab_size, on_value=1.0, off_value=0.0)

  log_probs = nn.log_softmax(logits)
  per_example_loss = -jnp.sum(log_probs * one_hot_labels, axis=-1)

  numerator = jnp.sum(label_weights * per_example_loss)
  denominator = jnp.sum(label_weights) + 1e-5
  loss = numerator / denominator
  return loss, per_example_loss, log_probs


def get_pretrain_loss(labels, lm_output, sentence_output):
  """Calculate loss for pretrain task.

  Args:
    labels: (masked_lm_ids, masked_lm_weights, next sentence_labels)
    lm_output: masked_lm layer output
    sentence_output: classification layer output

  Returns:
    mean loss across all local devices
  """
  masked_lm_ids, masked_lm_weights, next_sentence_labels = labels
  masked_label_loss, _, _ = get_masked_lm_output(
      lm_output, masked_lm_ids, label_weights=masked_lm_weights)

  sentence_loss, _ = compute_weighted_cross_entropy(sentence_output,
                                                    next_sentence_labels[:, 0])

  total_loss = masked_label_loss + sentence_loss
  return total_loss, masked_label_loss, sentence_loss


def get_pretrain_model():
  """Get pretrain model with pretrained weights loaded.

  Returns:
    jax_model: pretrain model with TF pretrained weights loaded to its
    transformer encoder part
  """
  # Get pretrained TF model configuration and variables from checkpoint
  if FLAGS.load_tf_weights:
    if FLAGS.load_mlperf_weights:
      tf_config, tf_vars, _ = utils.get_mlperf_model_variables(
          FLAGS.bert_config_file, FLAGS.init_checkpoint)
    else:
      tf_config, tf_vars, _ = utils.get_tf_model_variables(
          FLAGS.bert_config_file, FLAGS.init_checkpoint)
  else:
    tf_config = utils.get_tf_config(FLAGS.bert_config_file)

  # Generate JAX model using same model configuration as TF model
  if FLAGS.seed:
    seed = FLAGS.seed
  else:
    seed = np.int64(time.time())
  logging.info('RNG seed is %d.', seed)
  rng = random.PRNGKey(seed)
  tf.random.set_seed(seed)

  sequence_length = FLAGS.max_seq_length
  device_batch_size = int(FLAGS.train_batch_size // jax.device_count())
  model_kwargs = utils.convert_tf_config_to_jax_bert(tf_config)
  model_kwargs['num_token_predictions'] = FLAGS.max_predictions_per_seq
  model_kwargs['num_classes'] = 2
  with nn.stochastic(rng):
    model_def = bert_models.PretrainModel.partial(**model_kwargs)
    input_shape = (device_batch_size, sequence_length)
    inputs = jax.numpy.zeros(input_shape, dtype=jnp.int32)
    _, jax_model = model_def.create(rng, [inputs] * 4)
  # Update transformer encoder parameters with TF model pretrained weights
  if FLAGS.load_tf_weights:
    if FLAGS.load_mlperf_weights:
      jax_transformer_vars = utils.convert_mlperf_param_dict_to_jax(
          tf_vars, model_kwargs['emb_dim'], model_kwargs['num_heads'])
      jax_model.params.update(jax_transformer_vars)
    else:
      raise NotImplementedError(
          'Loading kerasBERT checkpoint for pretraining not supported yet.')
  else:
    encoder_vars = jax_model.params['transformer_encoder']
    encoder_vars['self_attention_mask'] = 0.0
    masked_lm_vars = jax_model.params['masked_lm']
    masked_lm_vars['0'] = 0.0
    masked_lm_vars['GatherIndexes_0'] = 0.0
    jax_model.params.update({'transformer_encoder': encoder_vars})
    jax_model.params.update({'masked_lm': masked_lm_vars})

  return jax_model, model_kwargs


def create_optimizer(model, model_kwargs, learning_rate=1e-4):
  """Create optimizer used for training model.

  MultiOpt is used to apply Adam/LAMB Optimizer with weight decay to all
  parameters except layer_norm and bias and Adam/LAMB Optimizer without weight
  decay for layer_norm and bias params.

  Args:
    model: JAX model to add optimizer to
    model_kwargs: Bert model config parameter dictionary.
    learning_rate: base learning rate used for initializing optimizer

  Returns:
    optimizer: model with Adam/LAMB Optimizer to be used for training
  """
  if FLAGS.use_lamb:
    weight_decay_def = bert_lamb.BertLAMB(
        learning_rate=learning_rate,
        beta1=FLAGS.lamb_beta_1, beta2=FLAGS.lamb_beta_2,
        eps=10**FLAGS.log_epsilon,
        weight_decay=FLAGS.lamb_weight_decay,
        num_layers=model_kwargs['num_layers'])
    no_decay_def = bert_lamb.BertLAMB(
        learning_rate=learning_rate,
        beta1=FLAGS.lamb_beta_1, beta2=FLAGS.lamb_beta_2,
        eps=10**FLAGS.log_epsilon, weight_decay=0.0,
        num_layers=model_kwargs['num_layers'])
  else:
    weight_decay_def = optim.Adam(
        learning_rate=learning_rate, eps=1e-6, weight_decay=FLAGS.lamb_weight_decay)
    no_decay_def = optim.Adam(
        learning_rate=learning_rate, eps=1e-6, weight_decay=0.0)

  def filter_weight_decay(key, _):
    return 'layer_norm' not in key and 'bias' not in key and 'layernorm' not in key

  def filter_other(key, _):
    return 'layer_norm' in key or 'bias' in key or 'layernorm' in key

  weight_decay_traversal = optim.ModelParamTraversal(filter_weight_decay)
  no_decay_traversal = optim.ModelParamTraversal(filter_other)
  optimizer_def = optim.MultiOptimizer(
      (weight_decay_traversal, weight_decay_def),
      (no_decay_traversal, no_decay_def))

  optimizer = optimizer_def.create(model)
  optimizer = jax_utils.replicate(optimizer)
  del model
  return optimizer


def create_learning_rate_scheduler(base_learning_rate=0.5,
                                   warmup_steps=1000,
                                   total_training_steps=100000,
                                   poly_power=1.0,
                                   start_warmup_step=0):
  """Create scheduler for learning rate."""

  # Implements linear warmup. I.e., if step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(step - start_warmup_step)/num_warmup_steps * init_lr`.
  def step_fn(step):
    """Step to learning rate function."""
    lr = base_learning_rate
    warmup_lr_fn = lambda x: x * ((step - start_warmup_step) / warmup_steps)
    poly_lr_fn = lambda x: x * (1 - step / total_training_steps)**poly_power
    return lax.cond((step - start_warmup_step) < warmup_steps, lr, warmup_lr_fn,
                    lr, poly_lr_fn)

  return step_fn


def train_step(optimizer, inputs, labels, learning_rate_fn, dropout_rng=None):
  """A single training step.

  Args:
    optimizer: optimizer used for training
    inputs: inputs to the model [word_ids, mask, type_ids]
    labels: target output [start_positions, end_positions]
    learning_rate_fn: function for tuning learning rate
    dropout_rng: random seed used for dropout

  Returns:
    new_optimizer: updated model optimizer after training step
    loss: sparse categorical crossentropy
    new_dropout_rng: new random seed to be used for next step
  """
  dropout_rng, new_dropout_rng = random.split(dropout_rng)

  def loss_fn(model):
    with nn.stochastic(dropout_rng):
      use_bf16 = FLAGS.use_bfloat16_activation
      dtype = jnp.bfloat16 if use_bf16 else jnp.float32
      lm_outputs, sentence_outputs = model(
          inputs, train=True, dtype=dtype)
      assert lm_outputs.dtype == jnp.float32
      assert sentence_outputs.dtype == jnp.float32
    total_loss, lm_loss, sentence_loss = get_pretrain_loss(
        labels, lm_outputs, sentence_outputs)
    return total_loss, (lm_loss, sentence_loss)

  def clip_by_global_normal(grads):
    _, treedef = jax.tree_flatten(grads)
    grads_flat = treedef.flatten_up_to(grads)
    grad_norms = [jnp.linalg.norm(gd)**2 for gd in grads_flat]
    global_norm = jnp.sqrt(jnp.sum(grad_norms))
    clip_norm = 1.0
    grads_flat = [
        gd * clip_norm / jnp.maximum(global_norm, clip_norm)
        for gd in grads_flat
    ]
    return jax.tree_unflatten(treedef, grads_flat)

  step = optimizer.state[0].step
  lr = learning_rate_fn(step)
  total_loss, (lm_loss,
               sentence_loss), grads = optimizer.compute_gradient(loss_fn)
  clipped_grads = clip_by_global_normal(grads)
  if FLAGS.reduce_gradients_in_bf16:
    clipped_grads = jax.tree_map(lambda x: x.astype(jnp.bfloat16),
                                 clipped_grads)
  clipped_grads = lax.psum(clipped_grads, 'batch')
  if FLAGS.reduce_gradients_in_bf16:
    clipped_grads = jax.tree_map(lambda x: x.astype(jnp.float32), clipped_grads)
  new_optimizer = optimizer.apply_gradient(clipped_grads, learning_rate=lr)

  return new_optimizer, total_loss, lm_loss, sentence_loss, new_dropout_rng


def empty_metrics():
  metrics = {'masked_lm_weighted_correct': 0.0, 'masked_lm_weighted_count': 0.0}
  local_device_count = jax.local_device_count()
  def broadcast(x):
    return np.reshape(np.broadcast_to(x, [local_device_count]),
                      [local_device_count, 1])
  return jax.tree_map(broadcast, metrics)


@partial(jax.pmap, axis_name='batch')
def allreduce_metrics(metrics):
  return lax.psum(metrics, axis_name='batch')


def eval_step(model, inputs, prev_metrics):
  """A single eval step."""
  input_ids = inputs['input_ids']
  input_mask = inputs['input_mask']
  segment_ids = inputs['segment_ids']
  mask_lm_positions = inputs['masked_lm_positions']

  masked_lm_ids = inputs['masked_lm_ids']
  masked_lm_weights = inputs['masked_lm_weights']
  use_bf16 = FLAGS.use_bfloat16_activation
  dtype = jnp.bfloat16 if use_bf16 else jnp.float32
  lm_outputs, _ = model([input_ids, input_mask, segment_ids, mask_lm_positions],
                        train=False, dtype=dtype)
  assert lm_outputs.dtype == jnp.float32
  _, masked_lm_example_loss, masked_lm_log_probs = get_masked_lm_output(
      lm_outputs, masked_lm_ids, label_weights=masked_lm_weights)
  masked_lm_log_probs = jnp.reshape(masked_lm_log_probs,
                                    (-1, masked_lm_log_probs.shape[-1]))
  masked_lm_predictions = jnp.argmax(masked_lm_log_probs, axis=-1)
  masked_lm_example_loss = jnp.reshape(masked_lm_example_loss, (-1))
  masked_lm_ids = jnp.reshape(masked_lm_ids, (-1))
  masked_lm_weights = jnp.reshape(masked_lm_weights, (-1))

  masked_lm_weighted_correct = jnp.multiply(
      lax.convert_element_type(
          jnp.equal(masked_lm_ids, masked_lm_predictions), jnp.float32),
      masked_lm_weights)
  masked_lm_weighted_correct = jnp.sum(masked_lm_weighted_correct)
  masked_lm_weighted_count = jnp.sum(masked_lm_weights)

  metrics = {
      'masked_lm_weighted_correct':
          jnp.reshape(masked_lm_weighted_correct, (-1)),
      'masked_lm_weighted_count':
          jnp.reshape(masked_lm_weighted_count, (-1))
  }

  return jax.tree_multimap(jnp.add, prev_metrics, metrics)


def get_masked_lm_accuracy(metrics):
  return (np.sum(metrics['masked_lm_weighted_correct']) /
          np.sum(metrics['masked_lm_weighted_count']))


def _write_metrics(eval_metrics, train_metrics, host_step, total_training_steps,
                   host_id):
  """Logs the accuracy metrics."""
  del host_id
  global RUN_STOP
  global TOTAL_STEPS
  if RUN_STOP:
    return

  eval_metrics = jax.tree_map(jax.device_get, eval_metrics)
  train_metrics = jax.tree_map(jax.device_get, train_metrics)

  masked_lm_accuracy = (
      np.sum(eval_metrics['masked_lm_weighted_correct']) /
      np.sum(eval_metrics['masked_lm_weighted_count']))
  total_loss = np.mean(train_metrics['total_loss'])
  lm_loss = np.mean(train_metrics['lm_loss'])
  sentence_loss = np.mean(train_metrics['sentence_loss'])

  mlp_log.mlperf_print('eval_accuracy', float(masked_lm_accuracy),
                       metadata={'epoch_num': host_step})

  logging.info('(Step %s / %s), masked_lm_accuracy: %s', host_step,
               total_training_steps, masked_lm_accuracy)
  logging.info(
      '(----Step %s / %s) Total loss: %s | LM loss: %s | Sentence loss: %s',
      host_step, total_training_steps, total_loss, lm_loss, sentence_loss)
  mlp_log.mlperf_print('eval_stop', None, metadata={'epoch_num': host_step})

  if masked_lm_accuracy >= FLAGS.target_accuracy:
    mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
    RUN_STOP = time.time()
    TOTAL_STEPS = host_step


def run_pretrain(optimizer):
  """Run bert pretraining.

  Args:
    optimizer: BERT model with pretraining layer

  Returns:
    optimizer: trained model
  """
  result_stats = {}
  def get_input_context():

    class InputContext():

      def __init__(self):
        self.input_pipeline_id = jax.host_id()
        self.num_input_pipelines = jax.host_count()
    return InputContext()

  summary_thread = thread.ThreadPoolExecutor(1, 'summary')
  host_id = jax.host_id()
  # Get input dataset
  input_files = []
  for input_pattern in FLAGS.input_files.split(','):
    input_files.extend(tf.io.gfile.glob(input_pattern))
  logging.info('*** Input Files ***')
  for input_file in input_files:
    logging.info('  %s', input_file)

  eval_input_files = []
  for input_pattern in FLAGS.eval_input_files.split(','):
    eval_input_files.extend(tf.io.gfile.glob(input_pattern))
  logging.info('*** Eval Input Files ***')
  for input_file in eval_input_files:
    logging.info('  %s', input_file)

  train_input_fn = input_pipeline.input_fn_builder(
      input_files=input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=True,
      num_cpu_threads=8)

  host_train_batch_size = FLAGS.train_batch_size // jax.host_count()
  host_eval_batch_size = FLAGS.eval_batch_size // jax.host_count()

  params = {'batch_size': host_train_batch_size}
  input_context = get_input_context()
  train_dataset = train_input_fn(params, input_context)
  train_iterator = iter(train_dataset)

  eval_input_fn = input_pipeline.input_fn_builder(
      input_files=eval_input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=False,
      num_cpu_threads=8,
      global_input_size=FLAGS.eval_sample_size)
  eval_params = {'batch_size': host_eval_batch_size}
  eval_dataset = eval_input_fn(eval_params, input_context)
  eval_iterator = iter(eval_dataset)

  # train step
  total_training_steps = FLAGS.total_training_steps
  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps,
      total_training_steps=FLAGS.total_training_steps,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step)

  # Device training loop cond.
  def device_train_loop_cond(args):
    _, _, _, _, _, _, step, epoch, num_steps_per_epoch = args
    return step // num_steps_per_epoch == epoch

  # Device training loop body.
  def device_train_loop_body(args):
    """Device training loop body."""
    (optimizer, total_loss, lm_loss, sentence_loss, new_dropout_rng, token,
     step, epoch, num_steps_per_epoch) = args
    device_batch_size = FLAGS.train_batch_size // jax.device_count()
    input_shape = [device_batch_size, FLAGS.max_seq_length]
    input_shape_pred = [device_batch_size, FLAGS.max_predictions_per_seq]
    (input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids,
     masked_lm_weights, next_sentence_labels), token = lax.infeed(
         token,
         shape=(jax.ShapedArray(input_shape, jnp.int32),
                jax.ShapedArray(input_shape, jnp.int32),
                jax.ShapedArray(input_shape, jnp.int32),
                jax.ShapedArray(input_shape_pred, jnp.int32),
                jax.ShapedArray(input_shape_pred, jnp.int32),
                jax.ShapedArray(input_shape_pred, jnp.float32),
                jax.ShapedArray([device_batch_size, 1], jnp.int32)))
    inputs = [input_ids, input_mask, segment_ids, masked_lm_positions]
    labels = [masked_lm_ids, masked_lm_weights, next_sentence_labels]
    optimizer, total_loss, lm_loss, sentence_loss, new_dropout_rng = train_step(
        optimizer,
        inputs,
        labels,
        learning_rate_fn,
        dropout_rng=new_dropout_rng)
    step += 1
    return (optimizer, total_loss, lm_loss, sentence_loss,
            new_dropout_rng, token, step, epoch, num_steps_per_epoch)

  # Device training loop.
  def device_train_loop(optimizer, dropout_rng, total_loss, lm_loss,
                        sentence_loss, step, epoch, num_steps_per_epoch):
    """Device training loop."""
    token = lax.create_token(step)
    (optimizer, total_loss, lm_loss, sentence_loss, dropout_rng,
     _, step, epoch, num_steps_per_epoch) = lax.while_loop(
         device_train_loop_cond, device_train_loop_body,
         (optimizer, total_loss, lm_loss, sentence_loss, dropout_rng, token,
          step, epoch, num_steps_per_epoch))
    return optimizer, total_loss, lm_loss, sentence_loss, dropout_rng, step

  if FLAGS.infeed:
    pmap_fn = jax.pmap
    if FLAGS.enable_buffer_donation:
      pmap_fn = functools.partial(pmap_fn, donate_argnums=(0, 1))
    if FLAGS.enable_wus:
      pmap_fn = functools.partial(
          pmap_fn, in_axes=(None, 0, None, None, None, None, None, None))

    p_train_epoch = pmap_fn(device_train_loop, axis_name='batch')
  else:
    # without infeed.
    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn),
        axis_name='batch')

  if FLAGS.infeed:
    # Infeed is currently synchronous, so do it in a background thread too
    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')

  pmap_fn = jax.pmap
  # Weight update sharding is not implemented yet for host train loop.
  # Enable wus on eval only if device loop is used.
  if FLAGS.enable_wus and FLAGS.infeed:
    pmap_fn = functools.partial(pmap_fn, in_axes=(None, 0, 0))
  p_eval_step = pmap_fn(eval_step, axis_name='batch')

  rng = random.PRNGKey(0)
  device_count = jax.local_device_count()
  dropout_rngs = random.split(rng, device_count)
  num_steps_per_epoch = np.int32(FLAGS.num_steps_per_epoch)
  if FLAGS.precompile:
    if FLAGS.infeed:
      if FLAGS.enable_wus:
        total_loss = np.float32(0.0)
        lm_loss = np.float32(0.0)
        sentence_loss = np.float32(0.0)
        host_step = 0
        host_epoch = 1
        optimizer = unbroadcast(optimizer)
        # the device training loop condition will immediately be false
        optimizer, total_loss, lm_loss, sentence_loss, _, _ = p_train_epoch(
            optimizer, dropout_rngs, total_loss, lm_loss, sentence_loss,
            host_step, host_epoch, num_steps_per_epoch)
      else:
        total_loss = jax_utils.replicate(np.float32(0.0))
        lm_loss = jax_utils.replicate(np.float32(0.0))
        sentence_loss = jax_utils.replicate(np.float32(0.0))
        device_step = jax_utils.replicate(0)
        device_epoch = jax_utils.replicate(1)
        # the device training loop condition will immediately be false
        optimizer, total_loss, lm_loss, sentence_loss, _, _ = p_train_epoch(
            optimizer, dropout_rngs, total_loss, lm_loss, sentence_loss,
            device_step, device_epoch, jax_utils.replicate(num_steps_per_epoch))

    else:
      train_input_shape = (host_train_batch_size, FLAGS.max_seq_length)
      train_input_shape_pred = (host_train_batch_size,
                                FLAGS.max_predictions_per_seq)
      word_id_data = jax.random.randint(rng, train_input_shape, 0, 10)
      mask_data = jax.random.randint(rng, train_input_shape, 0, 1)
      type_id_data = jax.random.randint(rng, train_input_shape, 0, 3)
      lm_mask = jax.random.randint(rng, train_input_shape_pred, 0, 5)
      masked_lm_ids = jax.random.randint(rng, train_input_shape_pred, 0, 2)
      masked_lm_weights = jax.random.randint(rng, train_input_shape_pred, 1,
                                             1).astype(np.float32)
      next_sentence_labels = jax.random.randint(rng, (host_train_batch_size, 1),
                                                0, 1)

      labels = [masked_lm_ids, masked_lm_weights, next_sentence_labels]
      train_inputs = [word_id_data, mask_data, type_id_data, lm_mask]
      train_inputs = common_utils.shard(train_inputs)
      labels = common_utils.shard(labels)
      p_train_step(optimizer, train_inputs, labels, dropout_rng=dropout_rngs)

    eval_input_shape = (host_eval_batch_size, FLAGS.max_seq_length)
    eval_input_shape_pred = (host_eval_batch_size,
                             FLAGS.max_predictions_per_seq)
    word_id_data = jax.random.randint(rng, eval_input_shape, 0, 10)
    mask_data = jax.random.randint(rng, eval_input_shape, 0, 1)
    type_id_data = jax.random.randint(rng, eval_input_shape, 0, 3)
    lm_mask = jax.random.randint(rng, eval_input_shape_pred, 0, 5)
    masked_lm_ids = jax.random.randint(rng, eval_input_shape_pred, 0, 2)
    masked_lm_weights = jax.random.randint(
        rng, eval_input_shape_pred, 1, 1).astype(np.float32)
    next_sentence_labels = jax.random.randint(rng, (host_eval_batch_size, 1), 0,
                                              1)

    eval_inputs = {
        'input_ids': word_id_data,
        'input_mask': mask_data,
        'segment_ids': type_id_data,
        'masked_lm_positions': lm_mask,
        'masked_lm_ids': masked_lm_ids,
        'masked_lm_weights': masked_lm_weights,
        'next_sentence_labels': next_sentence_labels
    }

    eval_inputs = common_utils.shard(eval_inputs)
    metrics = empty_metrics()
    optimizer_target = optimizer.target
    # Weight update sharding is not implemented yet for host train loop.
    # Enable wus on eval only if device loop is used.
    if FLAGS.enable_wus and FLAGS.infeed:
      optimizer_target = unbroadcast(optimizer_target)
    metrics = p_eval_step(optimizer_target, eval_inputs, metrics)
    metrics = allreduce_metrics(metrics)
  metrics = empty_metrics()
  time.sleep(FLAGS.init_sleep)
  allreduce_metrics(metrics)['masked_lm_weighted_correct'].block_until_ready()
  mlp_log.mlperf_print('init_stop', None)
  mlp_log.mlperf_print('run_start', None)
  # To make the logging consistent with other mlperf models,
  # in all the mlp_log, epochs are steps, and examples are sequences.
  mlp_log.mlperf_print('train_samples',
                       FLAGS.total_training_steps * FLAGS.train_batch_size)
  mlp_log.mlperf_print('eval_samples', FLAGS.eval_sample_size)
  xprof = None
  run_start = time.time()
  global RUN_STOP
  global TOTAL_STEPS
  RUN_STOP = False
  TOTAL_STEPS = False

  if host_id == 0:
    if FLAGS.end_to_end_profile:
      xprof = xprof_session.XprofSession()
      xprof.start_session(device_name='REDACTED',
                          enable_python_tracer=True,
                          host_trace_level=2)
    elif FLAGS.profile:
      profile_with_xprof_on_background(start_after_sec=FLAGS.profile_latency,
                                       profile_time_sec=FLAGS.profile_duration)

  if FLAGS.infeed:
    h_total_loss = np.float32(0.0)
    h_lm_loss = np.float32(0.0)
    h_sentence_loss = np.float32(0.0)

    d_total_loss = jax_utils.replicate(np.float32(0.0))
    d_lm_loss = jax_utils.replicate(np.float32(0.0))
    d_sentence_loss = jax_utils.replicate(np.float32(0.0))

  host_step, device_step = 0, jax_utils.replicate(0)
  device_epoch = jax_utils.replicate(0)
  num_train_epochs = FLAGS.total_training_steps // FLAGS.num_steps_per_epoch
  steps_per_epoch = num_steps_per_epoch
  if num_train_epochs >= 6:
    # Merge the first 6 epochs, as we do not have to do eval.
    steps_per_epoch = np.int32(num_steps_per_epoch * 6)
  for host_epoch in range(num_train_epochs):
    block_step = host_step
    # While BERT pretraining does not have epochs,
    # to make the logging consistent with other mlperf models,
    # in all the mlp_log, epochs are steps, and examples are sequences.
    mlp_log.mlperf_print(
        'block_start',
        None,
        metadata={
            'first_epoch_num': block_step,
            'epoch_count': FLAGS.num_steps_per_epoch
        })

    if not (num_train_epochs >= 6 and
            host_epoch in (1, 2, 3, 4, 5)) and FLAGS.infeed:
      if FLAGS.enable_wus:
        optimizer = unbroadcast(optimizer)
        (optimizer, total_loss, lm_loss, sentence_loss, dropout_rngs,
         device_step) = p_train_epoch(optimizer, dropout_rngs,
                                      h_total_loss, h_lm_loss, h_sentence_loss,
                                      host_step, host_epoch, steps_per_epoch)
      else:
        device_epoch = jax_utils.replicate(host_epoch)
        device_steps_per_epoch = jax_utils.replicate(steps_per_epoch)

        (optimizer, total_loss, lm_loss, sentence_loss, dropout_rngs,
         device_step) = p_train_epoch(optimizer, dropout_rngs,
                                      d_total_loss, d_lm_loss, d_sentence_loss,
                                      device_step, device_epoch,
                                      device_steps_per_epoch)
    # After first epoch, reduce the steps per epoch back to normal number.
    steps_per_epoch = num_steps_per_epoch

    # Training for one epoch.
    while int(host_step // FLAGS.num_steps_per_epoch) == host_epoch:
      input_data = next(train_iterator)
      input_data = jax.tree_map(lambda x: x.numpy(), input_data)
      input_data = jax.tree_map(common_utils.shard, input_data)
      input_ids = input_data['input_ids']
      input_mask = input_data['input_mask']
      segment_ids = input_data['segment_ids']
      masked_lm_positions = input_data['masked_lm_positions']
      masked_lm_ids = input_data['masked_lm_ids']
      masked_lm_weights = input_data['masked_lm_weights']
      next_sentence_labels = input_data['next_sentence_labels']

      # Infeed data to infeed queue.
      if FLAGS.infeed:
        for i, device in enumerate(jax.local_devices()):
          infeed_pool.submit(
              partial(device.transfer_to_infeed,
                      (input_ids[i], input_mask[i], segment_ids[i],
                       masked_lm_positions[i], masked_lm_ids[i],
                       masked_lm_weights[i], next_sentence_labels[i])))
      else:
        inputs = [input_ids, input_mask, segment_ids, masked_lm_positions]
        labels = [masked_lm_ids, masked_lm_weights, next_sentence_labels]
        (optimizer, total_loss, lm_loss, sentence_loss, dropout_rngs
         ) = p_train_step(optimizer, inputs, labels, dropout_rng=dropout_rngs)
      host_step += 1

    mlp_log.mlperf_print('block_stop', None, metadata={
        'first_epoch_num': block_step,
        'epoch_count': FLAGS.num_steps_per_epoch
    })
    # No need to do eval in the first 5 epochs as it has to traverse min 3M
    # samples.
    if host_epoch < 5:
      continue
    if host_step % FLAGS.num_steps_per_epoch == 0:
      mlp_log.mlperf_print(
          'eval_start', None, metadata={'epoch_num': host_step})
      optimizer_target = optimizer.target
      if FLAGS.enable_wus and FLAGS.infeed:
        optimizer_target = unbroadcast(optimizer_target)
      metrics = empty_metrics()
      for _ in range(FLAGS.max_eval_steps):
        inputs = jax.tree_map(lambda x: x.numpy(), next(eval_iterator))
        inputs = jax.tree_map(common_utils.shard, inputs)
        # Weight update sharding is not implemented yet for host train loop.
        # Enable wus on eval only if device loop is used.
        metrics = p_eval_step(optimizer_target, inputs, metrics)
      metrics = allreduce_metrics(metrics)
      train_metrics = {'total_loss': total_loss, 'lm_loss': lm_loss,
                       'sentence_loss': sentence_loss}
      # masked_lm_accuracy = get_masked_lm_accuracy(metrics)
      summary_thread.submit(partial(
          _write_metrics, metrics, train_metrics,
          host_step, total_training_steps, host_id))
    if host_step % FLAGS.num_steps_per_epoch == 0 and FLAGS.save_checkpoint:
      if host_id == 0:
        checkpoints.save_checkpoint(
            FLAGS.model_dir, optimizer, host_step, prefix='checkpoint', keep=1)
  allreduce_metrics(metrics)['masked_lm_weighted_correct'].block_until_ready()
  summary_thread.shutdown()
  if not RUN_STOP:
    mlp_log.mlperf_print('run_stop', None, metadata={'status': 'abort'})
  mlp_log.mlperf_print('run_final', None)

  if host_id == 0:
    if FLAGS.end_to_end_profile:
      xprof_url = xprof.end_session_and_get_url(tag='')
      logging.info('Xprof profile is at %s', xprof_url)


  if RUN_STOP:
    result_stats['total_time'] = RUN_STOP - run_start
    result_stats['total_steps'] = TOTAL_STEPS
  return optimizer, result_stats


def main(argv):
  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not gfile.IsDirectory(FLAGS.model_dir):
    gfile.MakeDirs(os.path.dirname(FLAGS.model_dir))

  logging.info('Number of recognized devices: %d', jax.local_device_count())
  logging.info('Import pretrained weights: %s', FLAGS.load_tf_weights)
  # Use hardware RNG for bernoulli randoms in dropout mask creation.
  if FLAGS.hardware_rng:
    set_hardware_bernoulli()
  num_success = 0
  runs = []
  for i in range(FLAGS.repeat_experiment):
    def run_exp():
      mlp_log.mlperf_print('cache_clear', None)
      mlp_log.mlperf_print('init_start', None)
      mlp_log.mlperf_print('global_batch_size', FLAGS.train_batch_size)
      mlp_log.mlperf_print('opt_learning_rate_warmup_steps', FLAGS.warmup_steps)
      mlp_log.mlperf_print('num_warmup_steps', FLAGS.warmup_steps)
      mlp_log.mlperf_print('start_warmup_step', FLAGS.start_warmup_step)
      mlp_log.mlperf_print('opt_lamb_weight_decay_rate', FLAGS.lamb_weight_decay)

      mlp_log.mlperf_print('max_sequence_length', FLAGS.max_seq_length)
      mlp_log.mlperf_print('opt_base_learning_rate', FLAGS.learning_rate)
      mlp_log.mlperf_print('opt_lamb_beta_1', FLAGS.lamb_beta_1)
      mlp_log.mlperf_print('opt_lamb_beta_2', FLAGS.lamb_beta_2)
      mlp_log.mlperf_print('opt_lamb_learning_rate_decay_poly_power', 1)
      mlp_log.mlperf_print('opt_gradient_accumulation_steps', 0)
      mlp_log.mlperf_print('max_predictions_per_seq',
                           FLAGS.max_predictions_per_seq)
      mlp_log.mlperf_print('opt_epsilon', 10**FLAGS.log_epsilon)
      mlp_log.mlperf_print('opt_learning_rate_training_steps',
                           FLAGS.total_training_steps)
      mlp_log.mlperf_print('submission_benchmark', 'bert')
      mlp_log.mlperf_print('submission_division', 'closed')
      mlp_log.mlperf_print('submission_org', 'google')
      mlp_log.mlperf_print('submission_platform',
                           'tpu-v3-%d' % jax.device_count())
      mlp_log.mlperf_print('submission_status', 'research')

      jax_model, model_kwargs = get_pretrain_model()
      optimizer = create_optimizer(jax_model, model_kwargs, learning_rate=None)
      _, result_stats = run_pretrain(optimizer)
      return result_stats
    result_stats = run_exp()
    if 'total_time' in result_stats:
      logging.info('Run %d/%d, total_time:%f, num_epochs:%f',
                   i, FLAGS.repeat_experiment, result_stats['total_time'],
                   result_stats['total_steps'])
      num_success += 1
      result_stats['RunID'] = i
      runs.append(result_stats)
  logging.info('Number of successfull runs:%d', num_success)
  for run_stat in runs:
    logging.info('Run %d/%d, total_time:%f, num_epochs:%f',
                 run_stat['RunID'], FLAGS.repeat_experiment,
                 run_stat['total_time'],
                 run_stat['total_steps'])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  app.run(main)
