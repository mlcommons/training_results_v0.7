"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools

from absl import app
from absl import flags
from absl import logging

from concurrent.futures import thread

from flax import jax_utils
from flax import nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import common_utils

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
# END GOOGLE-INTERNAL
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax.resnet34 import input_pipeline
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax.resnet34 import models

TARGET_ACCURACY = 0.759
FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=0.1,
    help='The base learning rate for the momentum optimizer.')

flags.DEFINE_float(
    'momentum', default=0.9,
    help='The decay rate (beta) used for the optimizer.')

flags.DEFINE_bool(
    'lars', default=None,
    help='Use LARS optimizer instead of Nesterov momentum.')

flags.DEFINE_integer(
    'batch_size', default=None,
    help='Batch size for training.')

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help='Coefficient for label smoothing at train time.')

flags.DEFINE_float(
    'weight_decay', default=0.0002,
    help='Coefficient for weight decay.')

flags.DEFINE_integer(
    'num_epochs', default=90,
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
    'bfloat16', default=True,
    help='Use bfloat16 precision instead of float32.')

flags.DEFINE_bool(
    'distributed_batchnorm', default=False,
    help='Use distributed batch normalization.')

flags.DEFINE_bool(
    'transpose_images', default=False,
    help='Use the "double transpose trick" for feeding images.')

flags.DEFINE_bool(
    'infeed', default=True,
    help='Stage out training loop to XLA using infeed for data loading.')

flags.DEFINE_bool(
    'precompile', default=True,
    help='Perform all XLA compilation before touching data.')

flags.DEFINE_string(
    'resnet_layers', default='34',
    help='Resnet layer string, 34, 50, e,g. SSD-34')
config.parse_flags_with_absl()


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(key, batch_size, image_size, model_dtype):
  """Model creation."""
  input_shape = (batch_size, image_size, image_size, 3)
  model_type = models.FakeResNet if FLAGS.fake_model else models.ResNet
  model_def = model_type.partial(
      num_classes=1000,
      axis_name='batch' if FLAGS.distributed_batchnorm else None,
      parameters={'dtype': model_dtype, 'conv0_space_to_depth': False},
      num_layers=FLAGS.resnet_layers)
  with nn.stateful() as init_state:
    _, model = model_def.create_by_shape(key, [(input_shape, model_dtype)])
  return model, init_state


def cross_entropy_loss(logits, labels, label_smoothing):
  num_classes = logits.shape[1]
  labels = common_utils.onehot(labels, num_classes)
  if label_smoothing > 0:
    labels = labels * (1 - label_smoothing) + label_smoothing / num_classes
  return -jnp.sum(labels * logits)


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels, label_smoothing=0)
  accuracy = jnp.sum(jnp.argmax(logits, -1) == labels)
  metrics = {
      'samples': logits.shape[0],
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


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
  """Polynomial learning rate fn."""
  if batch_size < 16384:
    base_lr = 10.0
    warmup_epochs = 5
  elif batch_size < 32768:
    base_lr = 25.0
    warmup_epochs = 5
  else:
    base_lr = 31.2
    warmup_epochs = 25
  def step_fn(step):
    current_epoch = step // steps_per_epoch + 1
    warmup_lr = base_lr * current_epoch / warmup_epochs
    warmup_steps = warmup_epochs * steps_per_epoch
    train_steps = num_epochs * steps_per_epoch
    poly_lr = base_lr * (
        1 - (step - warmup_steps) / (train_steps - warmup_steps + 1)) ** 2
    return jnp.where(current_epoch <= warmup_epochs, warmup_lr, poly_lr)
  return step_fn


def normalize_images(images):
  images -= jnp.array([[[input_pipeline.MEAN_RGB]]], dtype=images.dtype)
  images /= jnp.array([[[input_pipeline.STDDEV_RGB]]], dtype=images.dtype)
  return images


def train_step(optimizer, state, batch, prev_metrics, learning_rate_fn):
  """Single training step."""
  images, labels = batch['image'], batch['label']
  if FLAGS.transpose_images:
    images = jnp.transpose(images, [3, 0, 1, 2])
  images = normalize_images(images)
  if images.shape[1:] != (224, 224, 3):
    raise ValueError('images has shape {}'.format(images.shape))
  def loss_fn(model):
    with nn.stateful(state) as new_state:
      logits = model(images)
    loss = cross_entropy_loss(logits, labels, FLAGS.label_smoothing)
    return loss / logits.shape[0], (new_state, logits)

  lr = learning_rate_fn(optimizer.state[0].step)
  new_optimizer, _, (new_state, logits) = optimizer.optimize(
      loss_fn, learning_rate=lr)
  if FLAGS.train_metrics:
    metrics = compute_metrics(logits, labels)
    metrics = jax.tree_multimap(jnp.add, prev_metrics, metrics)
  else:
    metrics = {}
  return new_optimizer, new_state, metrics


def eval_step(model, state, batch, prev_metrics):
  images, labels = batch['image'], batch['label']
  if FLAGS.transpose_images:
    images = jnp.transpose(images, [3, 0, 1, 2])
  images = normalize_images(images)
  with nn.stateful(state, mutable=False):
    logits = model(images, train=False)
  metrics = compute_metrics(logits, labels)
  return jax.tree_multimap(jnp.add, prev_metrics, metrics)


def empty_metrics():
  metrics = {'samples': 0, 'loss': 0., 'accuracy': 0}
  local_device_count = jax.local_device_count()
  return jax.tree_map(lambda x: np.broadcast_to(x, [local_device_count]),
                      metrics)


@partial(jax.pmap, axis_name='batch')
def allreduce_metrics(metrics):
  return jax.tree_map(lambda x: lax.psum(x, axis_name='batch'), metrics)


def write_summary(summary_writer, device_metrics, prefix, epoch):
  """Writes summary metrics."""
  metrics = jax.tree_map(lambda x: jax.device_get(x[0]), device_metrics)
  samples = metrics.pop('samples')
  metrics = jax.tree_map(lambda x: x / samples, metrics)

  logging.info('%s epoch: %d, loss: %.4f, accuracy: %.2f',
               prefix, epoch, metrics['loss'], metrics['accuracy'] * 100)
  for key, val in metrics.items():
    tag = '{}_{}'.format(prefix, key)
    summary_writer.scalar(tag, val, epoch)
  summary_writer.flush()


def main(argv):
  del argv
  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL

  tf.enable_v2_behavior()

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(FLAGS.output_dir)
    # Write summaries in background thread to avoid blocking on device sync
    summary_thread = thread.ThreadPoolExecutor(1, 'summary')
  if FLAGS.infeed:
    # Infeed is currently synchronous, so do it in a background thread too
    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')

  rng = random.PRNGKey(0)

  image_size = 224

  batch_size = FLAGS.batch_size
  if batch_size is None:
    batch_size = min(128 * jax.device_count(), 32768)
  eval_batch_size = 128 * jax.device_count()
  local_batch_size = batch_size // jax.host_count()
  local_eval_batch_size = eval_batch_size // jax.host_count()
  device_batch_size = batch_size // jax.device_count()
  device_eval_batch_size = eval_batch_size // jax.device_count()
  device_last_eval_batch_size = (
      input_pipeline.EVAL_IMAGES % eval_batch_size) // jax.device_count()

  model_dtype = jnp.bfloat16 if FLAGS.bfloat16 else jnp.float32
  input_dtype = tf.bfloat16 if FLAGS.bfloat16 else tf.float32
  if FLAGS.transpose_images:
    train_input_shape = (224, 224, 3, device_batch_size)
    eval_input_shapes = [
        (224, 224, 3, bs) for bs in (device_eval_batch_size,
                                     device_last_eval_batch_size)]
  else:
    train_input_shape = (device_batch_size, 224, 224, 3)
    eval_input_shapes = [
        (bs, 224, 224, 3) for bs in (device_eval_batch_size,
                                     device_last_eval_batch_size)]

  num_epochs = FLAGS.num_epochs
  steps_per_epoch = input_pipeline.TRAIN_IMAGES / batch_size
  logging.info('steps_per_epoch: %f', steps_per_epoch)
  steps_per_eval = int(np.ceil(input_pipeline.EVAL_IMAGES / eval_batch_size))
  logging.info('steps_per_eval: %d', steps_per_eval)

  base_learning_rate = FLAGS.learning_rate * batch_size / 256.
  beta = FLAGS.momentum
  weight_decay = FLAGS.weight_decay

  logging.info('creating and initializing model and optimizer')
  model, state = create_model(rng, device_batch_size, image_size, model_dtype)
  state = jax_utils.replicate(state)
  if FLAGS.lars:
    weight_opt_def = optim.LARS(
        base_learning_rate, beta, weight_decay=weight_decay)
    other_opt_def = optim.Momentum(
        base_learning_rate, beta, weight_decay=0, nesterov=False)
    learning_rate_fn = polynomial_learning_rate_fn(
        batch_size, steps_per_epoch, num_epochs)
  else:
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
  optimizer = optimizer.replicate()
  del model  # do not keep a copy of the initial model

  p_train_step = jax.pmap(
      partial(train_step, learning_rate_fn=learning_rate_fn), axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  def device_train_loop_cond(args):
    _, _, _, _, step, epoch = args
    return step // steps_per_epoch == epoch
  def device_train_loop_body(args):
    optimizer, state, metrics, token, step, epoch = args
    (images, labels), token = lax.infeed(token, shape=(
        jax.ShapedArray(train_input_shape, model_dtype),
        jax.ShapedArray((device_batch_size,), jnp.int32)))
    batch = {'image': images, 'label': labels}
    optimizer, state, metrics = train_step(
        optimizer, state, batch, metrics, learning_rate_fn)
    step += 1
    return optimizer, state, metrics, token, step, epoch
  def device_train_loop(optimizer, state, metrics, step, epoch):
    token = lax.create_token(step)
    optimizer, state, metrics, _, step, _ = lax.while_loop(
        device_train_loop_cond,
        device_train_loop_body,
        (optimizer, state, metrics, token, step, epoch))
    return optimizer, state, metrics, step
  p_train_epoch = jax.pmap(device_train_loop, axis_name='batch')

  if FLAGS.precompile:
    logging.info('precompiling step/epoch functions')
    if FLAGS.infeed:
      # the device training loop condition will immediately be false
      p_train_epoch(optimizer, state, empty_metrics(),
                    jax_utils.replicate(0), jax_utils.replicate(1))
    else:
      batch = {'image': jnp.zeros((jax.local_device_count(),) +
                                  train_input_shape, model_dtype),
               'label': jnp.zeros((jax.local_device_count(),) +
                                  (device_batch_size,), jnp.int32)}
      p_train_step(optimizer, state, batch, empty_metrics())
    for dbs, eis in zip([device_eval_batch_size, device_last_eval_batch_size],
                        eval_input_shapes):
      batch = {'image': jnp.zeros((jax.local_device_count(),) +
                                  eis, model_dtype),
               'label': jnp.zeros((jax.local_device_count(),) +
                                  (dbs,), jnp.int32)}
      p_eval_step(optimizer.target, state, batch, empty_metrics())
    allreduce_metrics(empty_metrics())
    pmean = functools.partial(jax.lax.pmean, axis_name='batch')
    jax.pmap(pmean, axis_name='batch')(state)

  logging.info('constructing datasets')
  # pylint: disable=g-complex-comprehension
  train_ds, eval_ds = [
      input_pipeline.load_split(
          local_batch_size if train else local_eval_batch_size,
          image_size=image_size,
          dtype=input_dtype,
          train=train,
          transpose_images=FLAGS.transpose_images) for train in (True, False)
  ]
  # pylint: enable=g-complex-comprehension
  logging.info('constructing dataset iterators')
  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  logging.info('beginning training')
  host_step, device_step = 0, jax_utils.replicate(0)
  for epoch in range(num_epochs):
    device_epoch = jax_utils.replicate(epoch)
    metrics = empty_metrics()
    if FLAGS.infeed:
      optimizer, state, metrics, device_step = p_train_epoch(
          optimizer, state, metrics, device_step, device_epoch)
    while int(host_step // steps_per_epoch) == epoch:
      batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))  # pylint: disable=protected-access
      if FLAGS.infeed:
        for i, device in enumerate(jax.local_devices()):
          images, labels = batch['image'][i], batch['label'][i]
          assert images.shape == train_input_shape and labels.dtype == jnp.int32
          infeed_pool.submit(partial(
              device.transfer_to_infeed, (images, labels)))
      else:
        optimizer, state, metrics = p_train_step(
            optimizer, state, batch, metrics)
      host_step += 1
    if FLAGS.train_metrics:
      metrics = allreduce_metrics(metrics)
      if jax.host_id() == 0:
        summary_thread.submit(partial(
            write_summary, summary_writer, metrics, 'train', epoch + 1))
    if not FLAGS.distributed_batchnorm:  # otherwise it's already synced
      pmean = functools.partial(jax.lax.pmean, axis_name='batch')
      state = jax.pmap(pmean, axis_name='batch')(state)
    metrics = empty_metrics()
    for _ in range(steps_per_eval):
      batch = jax.tree_map(lambda x: x._numpy(), next(eval_iter))  # pylint: disable=protected-access
      metrics = p_eval_step(optimizer.target, state, batch, metrics)
    metrics = allreduce_metrics(metrics)
    if jax.host_id() == 0:
      summary_thread.submit(partial(
          write_summary, summary_writer, metrics, 'eval', epoch + 1))
    # TODO(deveci): do something like this from the summary thread:
    # if summary['accuracy'] > TARGET_ACCURACY:
    #   break
  if jax.host_id() == 0:
    summary_thread.shutdown()
  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()


if __name__ == '__main__':
  app.run(main)
