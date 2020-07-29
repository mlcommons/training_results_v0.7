"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

from concurrent.futures import thread
import math
import time

from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard

import jax
from jax import config
from jax import lax
from jax import random
import jax.numpy as jnp
from jax.util import partial

import numpy as np
import tensorflow.compat.v2 as tf

# BEGIN GOOGLE-INTERNAL
import REDACTED.learning.deepmind.xmanager2.client.google as xm
from REDACTED import xprof_session
# END GOOGLE-INTERNAL

from REDACTED.tensorflow_models.mlperf.models.rough.resnet_jax import mllog
from REDACTED.tensorflow_models.mlperf.models.rough.resnet_jax import mlperf_input_pipeline as input_pipeline
from REDACTED.tensorflow_models.mlperf.models.rough.resnet_jax import models

IMAGE_SIZE = 224
TARGET_ACCURACY = 0.759
FLAGS = flags.FLAGS

# A flag value of `None` or `-1` means we choose a value based on other flags.

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help='The base learning rate for the momentum optimizer.')

flags.DEFINE_float(
    'momentum', default=None,
    help='The decay rate (beta) used for the optimizer.')

flags.DEFINE_bool(
    'lars', default=True,
    help='Use LARS optimizer instead of Nesterov momentum.')

flags.DEFINE_integer(
    'batch_size', default=-1,
    help='Global batch size for training.')

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help='Coefficient for label smoothing at train time.')

flags.DEFINE_float(
    'weight_decay', default=None,
    help='Coefficient for weight decay.')

flags.DEFINE_integer(
    'num_epochs', default=None,
    help='Number of training epochs to use for learning rate schedule.')

flags.DEFINE_string(
    'output_dir', default=None,
    help='Directory to store model data.')

flags.DEFINE_bool(
    'train_metrics', default=True,
    help='Compute and log metrics during training.')

flags.DEFINE_bool(
    'fake_model', default=False,
    help='Use a tiny model (for debugging).')

flags.DEFINE_bool(
    'fake_data', default=False,
    help='Use fake data (for debugging).')

flags.DEFINE_bool(
    'bfloat16', default=True,
    help='Use bfloat16 precision instead of float32.')

flags.DEFINE_bool(
    'distributed_batchnorm', default=True,
    help='Use distributed batch normalization.')

flags.DEFINE_integer(
    'batchnorm_span', default=None,
    help='Number of examples for distributed batchnorm reduction.')

flags.DEFINE_string(
    'image_format', default=None,
    help='"NHWC", "HWCN" (TPU), or "HWNC" (TPU with space-to-depth).')

flags.DEFINE_bool(
    'space_to_depth', default=None,
    help='Use space-to-depth transformation for conv0.')

flags.DEFINE_bool(
    'device_loop', default=True,
    help='Stage out training loop to XLA.')

flags.DEFINE_integer(
    'epochs_per_loop', default=4,
    help='How many training epochs between each evaluation.')

flags.DEFINE_bool(
    'precompile', default=True,
    help='Perform all XLA compilation before touching data.')

flags.DEFINE_bool(
    'xprof', default=False,
    help='Collect Xprof traces on host 0.')

flags.DEFINE_integer(
    'seed', default=None,
    help='Random seed to use for initialization.')

config.parse_flags_with_absl()
mllogger = None
DONE = False


def local_replica_groups(inner_group_size):
  """Construct local nearest-neighbor rings given the JAX device assignment.

  For inner_group_size=8, each inner group is a tray with replica order:

  0/1 2/3
  7/6 5/4
  """
  world_size = jax.device_count()
  outer_group_size, ragged = divmod(world_size, inner_group_size)
  assert not ragged, 'inner group size must evenly divide global device count'
  # the last device should have maximal x and y coordinate
  def bounds_from_last_device(device):
    x, y, z = device.coords
    return (x + 1) * (device.core_on_chip + 1), (y + 1) * (z + 1)
  global_x, global_y = bounds_from_last_device(jax.devices()[-1])
  per_host_x, per_host_y = bounds_from_last_device(jax.local_devices(0)[-1])
  hosts_x, hosts_y = global_x // per_host_x, global_y // per_host_y
  assert inner_group_size in [2 ** i for i in range(1, 15)], \
      'inner group size must be a power of two'
  if inner_group_size <= 4:
    # inner group is Nx1 (core, chip, 2x1)
    inner_x, inner_y = inner_group_size, 1
    inner_perm = range(inner_group_size)
  else:
    if inner_group_size <= global_x * 2:
      # inner group is Nx2 (2x2 tray, 4x2 DF pod host, row of hosts)
      inner_x, inner_y = inner_group_size // 2, 2
    else:
      # inner group covers the full x dimension and must be >2 in y
      inner_x, inner_y = global_x, inner_group_size // global_x
    p = np.arange(inner_group_size)
    per_group_hosts_x = 1 if inner_x < per_host_x else inner_x // per_host_x
    p = p.reshape(inner_y // per_host_y, per_group_hosts_x,
                  per_host_y, inner_x // per_group_hosts_x)
    p = p.transpose(0, 2, 1, 3)
    p = p.reshape(inner_y // 2, 2, inner_x)
    p[:, 1, :] = p[:, 1, ::-1]
    inner_perm = p.reshape(-1)

  inner_replica_groups = [[o * inner_group_size + i for i in inner_perm]
                          for o in range(outer_group_size)]
  return inner_replica_groups


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def create_model(key, batch_size, image_size, model_dtype, space_to_depth):
  """Initialize a ResNet-50 model."""
  if space_to_depth:
    input_shape = (batch_size, image_size // 2, image_size // 2, 3 * 2 * 2)
  else:
    input_shape = (batch_size, image_size, image_size, 3)
  model_type = models.FakeResNet if FLAGS.fake_model else models.ResNet
  batchnorm_span = FLAGS.batchnorm_span
  if batchnorm_span is None:
    batchnorm_span = max(batch_size, 64)
  if FLAGS.distributed_batchnorm and (
      batch_size < batchnorm_span <= batch_size * jax.device_count()):
    mllogger.event('model_bn_span', batchnorm_span)
    model_def = model_type.partial(
        num_classes=1000,
        axis_name='batch',
        axis_index_groups=local_replica_groups(batchnorm_span // batch_size),
        dtype=model_dtype,
        conv0_space_to_depth=space_to_depth)
  else:
    mllogger.event('model_bn_span', batch_size)
    model_def = model_type.partial(num_classes=1000, dtype=model_dtype,
                                   conv0_space_to_depth=space_to_depth)
  with nn.stateful() as init_state:
    _, params = model_def.init_by_shape(key, [(input_shape, model_dtype)])
  model = nn.Model(model_def, params)
  return model, init_state


broadcast = jax_utils.replicate


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


def cross_entropy_loss(logits, labels, label_smoothing, mask=None):
  num_classes = logits.shape[1]
  labels = jax.nn.one_hot(labels, num_classes)
  if label_smoothing > 0:
    labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
  log_likelihoods = labels * logits
  if mask is not None:
    log_likelihoods = jnp.where(mask[:, None], log_likelihoods, 0)
  return -jnp.sum(log_likelihoods)


def compute_metrics(logits, labels):
  mask = (labels != -1)
  return {
      'samples': jnp.sum(mask),
      'loss': cross_entropy_loss(logits, labels, label_smoothing=0, mask=mask),
      'accuracy': jnp.sum(jnp.where(mask, jnp.argmax(logits, -1) == labels, 0)),
  }


def piecewise_constant(boundaries, values, t):
  index = jnp.sum(boundaries < t)
  return jnp.take(values, index)


def piecewise_learning_rate_fn(base_learning_rate, steps_per_epoch, num_epochs):
  boundaries = np.array([30, 60, 80]) * steps_per_epoch * num_epochs / 90
  values = np.array([1., 0.1, 0.01, 0.001]) * base_learning_rate
  def step_fn(step):
    lr = piecewise_constant(boundaries, values, step)
    lr = lr * jnp.minimum(1., step / 5. / steps_per_epoch)
    return lr
  return step_fn


def polynomial_learning_rate_fn(batch_size, steps_per_epoch, num_epochs):
  """Polynomial learning rate schedule for LARS optimizer."""
  if batch_size < 16384:
    base_lr = 10.0
    warmup_epochs = 5
  elif batch_size < 32768:
    base_lr = 25.0
    warmup_epochs = 5
  elif batch_size < 65536:
    base_lr = 29.0
    warmup_epochs = 18
  else:
    base_lr = 23.080977304670448
    warmup_epochs = 33
  warmup_steps = warmup_epochs * steps_per_epoch
  train_steps = num_epochs * steps_per_epoch
  decay_steps = train_steps - warmup_steps + 1
  end_lr = 0.0001
  def step_fn(step):
    warmup_lr = base_lr * step / warmup_steps
    decay_step = jnp.minimum(step - warmup_steps, decay_steps)
    poly_lr = end_lr + (base_lr - end_lr) * (1 - decay_step / decay_steps) ** 2
    return jnp.where(step <= warmup_steps, warmup_lr, poly_lr)
  mllogger.event('lars_opt_base_learning_rate', base_lr)
  mllogger.event('lars_opt_learning_rate_warmup_epochs', warmup_epochs)
  mllogger.event('lars_opt_end_learning_rate', end_lr)
  mllogger.event('lars_opt_learning_rate_decay_steps', decay_steps)
  mllogger.event('lars_opt_learning_rate_decay_poly_power', 2)
  return step_fn


def maybe_transpose_images(images, image_format):
  if image_format == 'NHWC':
    return images
  elif image_format == 'HWCN':
    return images.transpose([3, 0, 1, 2])
  elif image_format == 'HWNC':
    return images.transpose([2, 0, 1, 3])
  else:
    raise ValueError('unknown format: {}'.format(image_format))


def normalize_images(images, space_to_depth):
  mean = np.array([[[input_pipeline.MEAN_RGB]]])
  stddev = np.array([[[input_pipeline.STDDEV_RGB]]])
  if space_to_depth:
    mean = np.broadcast_to(mean, (1, 1, 1, 4, 3)).reshape(1, 1, 1, 12)
    stddev = np.broadcast_to(stddev, (1, 1, 1, 4, 3)).reshape(1, 1, 1, 12)
  images -= jnp.array(mean, dtype=images.dtype)
  images /= jnp.array(stddev, dtype=images.dtype)
  return images


def train_step(optimizer, state, batch, prev_metrics, learning_rate_fn,
               image_format, space_to_depth):
  """Return updated optimizer, batchnorm state, and metrics given a batch."""
  images, labels = batch
  images = maybe_transpose_images(images, image_format)
  images = normalize_images(images, space_to_depth)
  if (space_to_depth and images.shape[1:] != (112, 112, 12) or
      not space_to_depth and images.shape[1:] != (224, 224, 3)):
    raise ValueError('images has shape {}'.format(images.shape))
  def loss_fn(model):
    with nn.stateful(state) as new_state:
      logits = model(images)
    loss = cross_entropy_loss(logits, labels, FLAGS.label_smoothing)
    return loss / logits.shape[0], (new_state, logits)

  lr = learning_rate_fn(optimizer.state[0].step)
  (_, (new_state, logits)), grad = jax.value_and_grad(
      loss_fn, has_aux=True)(optimizer.target)
  grad = lax.pmean(grad, 'batch')
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
  if FLAGS.train_metrics:
    metrics = compute_metrics(logits, labels)
    metrics = jax.tree_multimap(jnp.add, prev_metrics, metrics)
  else:
    metrics = {}
  return new_optimizer, new_state, metrics


def sync_batchnorm_stats(state):
  # TODO(jekbradbury): use different formula for running variances?
  return lax.pmean(state, axis_name='batch')


def eval_step(model, state, batch, prev_metrics, image_format, space_to_depth):
  images, labels = batch
  images = maybe_transpose_images(images, image_format)
  images = normalize_images(images, space_to_depth)
  with nn.stateful(state, mutable=False):
    logits = model(images, train=False)
  metrics = compute_metrics(logits, labels)
  return jax.tree_multimap(jnp.add, prev_metrics, metrics)


def allreduce_metrics(metrics):
  return lax.psum(metrics, axis_name='batch')


def per_host_sum(x):
  return jax.lax.psum(x, 'hosts')


def per_host_sum_pmap(in_tree):
  """Execute sum on in_tree's leaves over ICI."""
  ldc = jax.local_device_count()
  host_psum = jax.pmap(per_host_sum, axis_name='hosts')
  def pre_pmap(x):
    y = np.zeros((ldc, *x.shape), dtype=x.dtype)
    y[0] = x
    return y
  def post_pmap(x):
    return jax.device_get(x)[0]
  return jax.tree_map(post_pmap, host_psum(jax.tree_map(pre_pmap, in_tree)))


def write_summary(summary_writer, device_metrics, prefix, epoch):
  global DONE
  metrics = jax.tree_map(lambda x: jax.device_get(x[0]), device_metrics)
  samples = metrics.pop('samples')
  metrics = jax.tree_map(lambda x: x / samples, metrics)
  logging.info('%s epoch: %d, loss: %.4f, accuracy: %.2f',
               prefix, epoch, metrics['loss'], metrics['accuracy'] * 100)
  if prefix == 'eval':
    if not DONE:
      mllogger.event('eval_accuracy', metrics['accuracy'],
                     metadata={'epoch_num': epoch})
      mllogger.end(
          'block_stop',
          metadata={'first_epoch_num': epoch - FLAGS.epochs_per_loop + 1,
                    'epoch_count': FLAGS.epochs_per_loop})
      if metrics['accuracy'] > TARGET_ACCURACY:
        mllogger.end('run_stop', metadata={'status': 'success'})
        DONE = True
    if not DONE:
      mllogger.start('block_start',
                     metadata={'first_epoch_num': epoch + 1,
                               'epoch_count': FLAGS.epochs_per_loop})
  for key, val in metrics.items():
    tag = '{}_{}'.format(prefix, key)
    summary_writer.scalar(tag, val, epoch)
  summary_writer.flush()


def init_mllogger():
  global mllogger
  mllogger = mllog.MLLogger(logging.get_absl_logger(), jax.host_id())


def main(argv):
  del argv
  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL

  tf.enable_v2_behavior()
  init_mllogger()

  mllogger.event('cache_clear')
  mllogger.start('init_start')
  mllogger.event('submission_org', 'Google')
  mllogger.event('submission_platform', 'TPUv3-{}'.format(jax.device_count()))
  mllogger.event('submission_division', 'closed')
  mllogger.event('submission_status', 'research')
  mllogger.event('submission_benchmark', 'resnet')
  mllogger.event('train_samples', input_pipeline.TRAIN_IMAGES)
  mllogger.event('eval_samples', input_pipeline.EVAL_IMAGES)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.output_dir)
    # Write summaries in background thread to avoid blocking on device sync
    summary_thread = thread.ThreadPoolExecutor(1, 'summary')
  # Infeed is currently synchronous, so do it in a background thread too
  infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')

  if FLAGS.seed is not None:
    seed = FLAGS.seed
  else:
    seed = np.uint32(time.time() if jax.host_id() == 0 else 0)
    seed = per_host_sum_pmap(seed)

  mllogger.event('seed', int(seed))
  key = random.PRNGKey(seed)

  batch_size = FLAGS.batch_size
  if batch_size == -1:
    if jax.device_count() > 4096:
      batch_size = 65536
    else:
      batch_size = min(128 * jax.device_count(), 32768)
  mllogger.event('global_batch_size', batch_size)
  eval_batch_size = min(input_pipeline.EVAL_IMAGES, 256 * jax.device_count())
  device_batch_size = batch_size // jax.device_count()
  device_eval_batch_size = int(math.ceil(eval_batch_size / jax.device_count()))

  model_dtype = jnp.bfloat16 if FLAGS.bfloat16 else jnp.float32
  input_dtype = tf.bfloat16 if FLAGS.bfloat16 else tf.float32

  num_epochs = FLAGS.num_epochs
  if num_epochs is None:
    if batch_size < 32768:
      num_epochs = 56
    elif batch_size < 65536:
      num_epochs = 64
    else:
      num_epochs = 92

  steps_per_epoch = input_pipeline.TRAIN_IMAGES / batch_size
  # match TF submission behavior (round steps per loop up)
  steps_per_loop = int(math.ceil(steps_per_epoch * FLAGS.epochs_per_loop))
  # also apply rounding loop up to next step to "epochs" in LR schedule
  steps_per_epoch *= steps_per_loop / (steps_per_epoch * FLAGS.epochs_per_loop)

  steps_per_eval = int(math.ceil(input_pipeline.EVAL_IMAGES / eval_batch_size))

  base_learning_rate = FLAGS.learning_rate * batch_size / 256.
  beta = FLAGS.momentum
  if beta is None:
    if batch_size < 32768:
      beta = 0.9
    elif batch_size < 65536:
      beta = 0.929
    else:
      beta = 0.9537213777059405
  weight_decay = FLAGS.weight_decay
  if weight_decay is None:
    weight_decay = 2e-4 if batch_size < 32768 else 1e-4

  space_to_depth = FLAGS.space_to_depth
  if space_to_depth is None:
    space_to_depth = device_batch_size <= 8

  image_format = FLAGS.image_format
  if image_format is None:
    if space_to_depth and device_batch_size <= 8:
      image_format = 'HWNC'
    else:
      image_format = 'HWCN'

  image_size = input_pipeline.IMAGE_SIZE
  if space_to_depth:
    train_input_shape = (
        device_batch_size, image_size // 2, image_size // 2, 12)
    eval_input_shape = (
        device_eval_batch_size, image_size // 2, image_size // 2, 12)
  else:
    train_input_shape = (device_batch_size, image_size, image_size, 3)
    eval_input_shape = (device_eval_batch_size, image_size, image_size, 3)
  if image_format == 'HWCN':
    train_input_shape = tuple(train_input_shape[i] for i in [1, 2, 3, 0])
    eval_input_shape = tuple(eval_input_shape[i] for i in [1, 2, 3, 0])
  elif image_format == 'HWNC':
    train_input_shape = tuple(train_input_shape[i] for i in [1, 2, 0, 3])
    eval_input_shape = tuple(eval_input_shape[i] for i in [1, 2, 0, 3])

  model, state = create_model(
      key, device_batch_size, image_size, model_dtype, space_to_depth)

  if FLAGS.lars:
    mllogger.event('opt_name', 'lars')
    mllogger.event('lars_opt_weight_decay', weight_decay)
    mllogger.event('lars_opt_momentum', beta)
    mllogger.event('lars_epsilon', 0)
    weight_opt_def = optim.LARS(
        base_learning_rate, beta, weight_decay=weight_decay)
    other_opt_def = optim.Momentum(
        base_learning_rate, beta, weight_decay=0, nesterov=False)
    learning_rate_fn = polynomial_learning_rate_fn(
        batch_size, steps_per_epoch, num_epochs)
  else:
    mllogger.event('opt_name', 'sgd')
    mllogger.event('sgd_opt_momentum', beta)
    weight_opt_def = optim.Momentum(
        base_learning_rate, beta, weight_decay=weight_decay, nesterov=True)
    other_opt_def = optim.Momentum(
        base_learning_rate, beta, weight_decay=0, nesterov=True)
    learning_rate_fn = piecewise_learning_rate_fn(
        base_learning_rate, steps_per_epoch, num_epochs)
  def filter_weights(key, _):
    return 'bias' not in key and 'scale' not in key
  def filter_other(key, _):
    return 'bias' in key or 'scale' in key
  weight_traversal = optim.ModelParamTraversal(filter_weights)
  other_traversal = optim.ModelParamTraversal(filter_other)
  optimizer_def = optim.MultiOptimizer((weight_traversal, weight_opt_def),
                                       (other_traversal, other_opt_def))
  optimizer = optimizer_def.create(model)
  del model  # do not keep a copy of the initial model

  optimizer = broadcast(optimizer)
  state = broadcast(state)
  empty_metrics = broadcast({'samples': 0, 'loss': 0., 'accuracy': 0})

  p_allreduce_metrics = jax.pmap(allreduce_metrics, axis_name='batch')

  p_sync_batchnorm_stats = jax.pmap(sync_batchnorm_stats, axis_name='batch')

  def host_loop_train_step(optimizer, state, metrics):
    token = lax.create_token(optimizer.state[0].step)
    batch, token = lax.infeed(token, shape=(
        jax.ShapedArray(train_input_shape, model_dtype),
        jax.ShapedArray((device_batch_size,), jnp.int32)))
    optimizer, state, metrics = train_step(optimizer, state, batch, metrics,
                                           learning_rate_fn, image_format,
                                           space_to_depth)
    return optimizer, state, metrics

  p_host_loop_train_step = jax.pmap(
      host_loop_train_step, axis_name='batch', in_axes=(None, 0, 0))

  def host_loop_eval_step(model, state, metrics):
    token = lax.create_token(metrics['samples'])
    batch, token = lax.infeed(token, shape=(
        jax.ShapedArray(eval_input_shape, model_dtype),
        jax.ShapedArray((device_eval_batch_size,), jnp.int32)))
    metrics = eval_step(
        model, state, batch, metrics, image_format, space_to_depth)
    return metrics

  p_host_loop_eval_step = jax.pmap(
      host_loop_eval_step, axis_name='batch', in_axes=(None, None, 0))

  def device_train_loop_cond(args):
    _, _, _, _, step, loop = args
    return step // steps_per_loop == loop
  def device_train_loop_body(args):
    optimizer, state, metrics, token, step, loop = args
    batch, token = lax.infeed(token, shape=(
        jax.ShapedArray(train_input_shape, model_dtype),
        jax.ShapedArray((device_batch_size,), jnp.int32)))
    optimizer, state, metrics = train_step(optimizer, state, batch, metrics,
                                           learning_rate_fn, image_format,
                                           space_to_depth)
    step += 1
    return optimizer, state, metrics, token, step, loop
  def device_train_loop(optimizer, state, metrics, step, loop):
    token = lax.create_token(step)
    optimizer, state, metrics, _, step, _ = lax.while_loop(
        device_train_loop_cond,
        device_train_loop_body,
        (optimizer, state, metrics, token, step, loop))
    state = sync_batchnorm_stats(state)
    metrics = allreduce_metrics(metrics)
    return optimizer, state, metrics, step

  p_train_loop = jax.pmap(
      device_train_loop, axis_name='batch', in_axes=(None, None, 0, None, None))

  # BEGIN GOOGLE-INTERNAL
  def maybe_start_xprof(seconds):
    if jax.host_id() == 0 and FLAGS.xprof:
      xprof = xprof_session.XprofSession()
      xprof.start_session('REDACTED', True, 2)
      def sleep_and_end_xprof():
        time.sleep(seconds)
        logging.info('Xprof URL: %s', xprof.end_session_and_get_url(
            tag='flax resnet, {} devices, batch {} per device'.format(
                jax.device_count(), device_batch_size)))
      thread.ThreadPoolExecutor(1, 'xprof').submit(sleep_and_end_xprof)
  # END GOOGLE-INTERNAL

  if FLAGS.precompile:
    logging.info('precompiling step/loop functions')
    if FLAGS.device_loop:
      # the device training loop condition will immediately be false
      p_train_loop(unbroadcast(optimizer), unbroadcast(state), empty_metrics,
                   jnp.array(0, dtype=jnp.int32), 1)
    else:
      for device in jax.local_devices():
        images = np.zeros(train_input_shape, model_dtype)
        labels = np.zeros((device_batch_size,), np.int32)
        infeed_pool.submit(partial(device.transfer_to_infeed, (images, labels)))
      p_host_loop_train_step(unbroadcast(optimizer), state, empty_metrics)
      p_sync_batchnorm_stats(state)
    for device in jax.local_devices():
      images = np.zeros(eval_input_shape, model_dtype)
      labels = np.zeros((device_eval_batch_size,), np.int32)
      infeed_pool.submit(partial(device.transfer_to_infeed, (images, labels)))
    p_host_loop_eval_step(
        unbroadcast(optimizer.target), unbroadcast(state), empty_metrics)
    p_allreduce_metrics(empty_metrics)['accuracy'].block_until_ready()
    logging.info('finished precompiling')

  # BEGIN GOOGLE-INTERNAL
  maybe_start_xprof(20)
  # END GOOGLE-INTERNAL
  if not FLAGS.fake_data:
    logging.info('constructing datasets')
    # pylint: disable=g-complex-comprehension
    train_ds, eval_ds = [
        input_pipeline.load_split(
            device_batch_size if train else device_eval_batch_size,
            dtype=input_dtype,
            train=train,
            image_format=image_format,
            space_to_depth=space_to_depth,
            cache_uncompressed=jax.device_count() > 64)
        for train in (True, False)]
    logging.info('constructing dataset iterators')
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

  local_devices = jax.local_devices()
  host_step, device_step = 0, broadcast(0)
  mllogger.end('init_stop')
  mllogger.start('run_start')
  mllogger.start('block_start',
                 metadata={'first_epoch_num': 1,
                           'epoch_count': FLAGS.epochs_per_loop})
  for loop in range(int(math.ceil(num_epochs / FLAGS.epochs_per_loop)) + 2):
    # BEGIN GOOGLE-INTERNAL
    if loop == 10: maybe_start_xprof(1)
    # END GOOGLE-INTERNAL
    metrics = empty_metrics
    if FLAGS.device_loop:
      optimizer, state, metrics, device_step = p_train_loop(
          unbroadcast(optimizer), unbroadcast(state), metrics,
          unbroadcast(device_step), loop)
    while int(host_step // steps_per_loop) == loop:
      if not FLAGS.device_loop:
        optimizer, state, metrics = p_host_loop_train_step(
            unbroadcast(optimizer), state, metrics)
      # pylint: disable=protected-access
      while infeed_pool._work_queue.qsize() > 100:
        time.sleep(0.01)
      for device in local_devices:
        if FLAGS.fake_data:
          images = np.zeros(train_input_shape, model_dtype)
          labels = np.zeros((device_batch_size,), np.int32)
        else:
          # pylint: disable=protected-access
          images, labels = jax.tree_map(lambda x: x._numpy(), next(train_iter))
        assert images.shape == train_input_shape and labels.dtype == jnp.int32
        infeed_pool.submit(partial(device.transfer_to_infeed, (images, labels)))
      host_step += 1
    epoch = (loop + 1) * FLAGS.epochs_per_loop
    if FLAGS.train_metrics:
      if not FLAGS.device_loop:
        metrics = p_allreduce_metrics(metrics)
      if jax.host_id() == 0:
        summary_thread.submit(partial(
            write_summary, summary_writer, metrics, 'train', epoch))
    if not FLAGS.device_loop:
      state = p_sync_batchnorm_stats(state)
    metrics = empty_metrics
    for _ in range(steps_per_eval):
      metrics = p_host_loop_eval_step(
          unbroadcast(optimizer.target), unbroadcast(state), metrics)
      for device in local_devices:
        if FLAGS.fake_data:
          images = np.zeros(eval_input_shape, model_dtype)
          labels = np.zeros((device_eval_batch_size,), np.int32)
        else:
          # pylint: disable=protected-access
          images, labels = jax.tree_map(lambda x: x._numpy(), next(eval_iter))
        assert images.shape == eval_input_shape and labels.dtype == jnp.int32, \
            'images.shape={}'.format(images.shape)
        infeed_pool.submit(partial(device.transfer_to_infeed, (images, labels)))
    metrics = p_allreduce_metrics(metrics)
    if jax.host_id() == 0:
      summary_thread.submit(partial(
          write_summary, summary_writer, metrics, 'eval', epoch))
  # Wait until computations are done before exiting
  p_allreduce_metrics(metrics)['accuracy'].block_until_ready()
  if jax.host_id() == 0:
    summary_thread.shutdown()
    if not DONE:
      mllogger.end('run_stop', metadata={'status': 'aborted'})


if __name__ == '__main__':
  app.run(main)
