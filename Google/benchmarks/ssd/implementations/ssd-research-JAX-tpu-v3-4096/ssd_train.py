# Lint as: python3
"""SSD main training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from concurrent.futures import thread
import math

import time
from absl import app
from absl import flags
from absl import logging

from flax import jax_utils
from flax import nn
from flax import optim

import jax

from jax import config
from jax import lax
from jax import random

from jax.interpreters import sharded_jit
import jax.numpy as jnp
from jax.util import partial
import numpy as onp

import tensorflow.compat.v2 as tf

import REDACTED.learning.deepmind.xmanager2.client.google as xm
from REDACTED import xprof_session
from REDACTED.tensorflow_models.mlperf.models.rough.mlp_log import mlp_log
from REDACTED.tensorflow_models.mlperf.models.rough.ssd import coco_metric
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import input_pipeline
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_architecture
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_constants
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_model
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_resnet_checkpoint_reader


flags.DEFINE_string(
    'resnet_checkpoint', None,
    'Location of the ResNet checkpoint to use for model '
    'initialization.')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')

flags.DEFINE_integer('global_batch_size', 1024, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 1024, 'evaluation batch size')
flags.DEFINE_integer('eval_samples', ssd_constants.EVAL_SAMPLES,
                     'The number of samples for evaluation.')
flags.DEFINE_integer('iterations_per_loop', 1000,
                     'Number of iterations per TPU training loop')
flags.DEFINE_string(
    'training_file_pattern',
    '/placer/prod/home/tpu-perf-team/mlperf/ssd/train*',
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string('validation_file_pattern',
                    '/placer/prod/home/tpu-perf-team/mlperf/ssd/coco_val*',
                    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')

flags.DEFINE_bool(
    'use_fake_data', False,
    'Use fake data to reduce the input preprocessing overhead (for unit tests)')

flags.DEFINE_string(
    'val_json_file',
    '/placer/prod/home/tpu-perf-team/mlperf/ssd/instances_val2017.json',
    'COCO validation JSON containing golden bounding boxes.')

flags.DEFINE_integer('num_examples_per_epoch', 118287,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 78, 'Number of epochs for training')
flags.DEFINE_multi_integer(
    'input_partition_dims',
    default=None,
    help=('Number of partitions on each dimension of the input. Each TPU core'
          ' processes a partition of the input image in parallel using spatial'
          ' partitioning.'))
flags.DEFINE_bool('run_cocoeval', True, 'Whether to run cocoeval')
flags.DEFINE_string(
    'model_dir',
    default=None,
    help=('The directory where the model and summaries are stored.'))
flags.DEFINE_bool('use_bfloat16', True, 'use bfloat16')

flags.DEFINE_integer('lr_warmup_epoch', 5, '')
flags.DEFINE_float(
    'base_learning_rate', default=ssd_constants.BASE_LEARNING_RATE,
    help='Base learning rate.')

flags.DEFINE_float('first_lr_drop_epoch', default=42.6, help='')
flags.DEFINE_float('second_lr_drop_epoch', default=53.3, help='')

flags.DEFINE_bool(
    'precompile', default=True,
    help='Perform all XLA compilation before touching data.')

flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Apply double transpose.')

# Used a different flag name, so we dont trigger the optimization in shared code
flags.DEFINE_bool(
    'conv0_space_to_depth', default=True,
    help='Space to Depth Optimization.')

flags.DEFINE_bool(
    'infeed', default=True,
    help='Stage out training loop to XLA using infeed for data loading.')

flags.DEFINE_bool(
    'profile', default=True,
    help='Enable programmatic profile with xprof.')

flags.DEFINE_bool(
    'detailed_time', default=False,
    help='Shows eval, train and coco_eval times separately(Adds barriers avoid '
    'in default mode)')

flags.DEFINE_bool(
    'enable_wus', default=True,
    help='Whether to enable weight update sharding')

flags.DEFINE_integer(
    'num_partitions', default=1, help=('Number of partitions in SPMD.'))

flags.DEFINE_integer(
    'bn_group_size', default=1, help=('Num. cores for Distributed Batch Norm.'))

flags.DEFINE_integer(
    'repeat_experiment', default=1, help=('Number of runs'))

flags.DEFINE_integer('seed', None, 'Random seed')
# Adds jax_log_compiles flag to print compilation logs on the jax side.
config.parse_flags_with_absl()
FLAGS = flags.FLAGS

coco_gt = None


# pylint: disable=g-complex-comprehension
def _tranposed_box_shapes(device_batch_size):
  return [jax.ShapedArray(shape, jnp.float32)
          for shape in ((device_batch_size, 38, 38, 4, 4),
                        (device_batch_size, 19, 19, 6, 4),
                        (device_batch_size, 10, 10, 6, 4),
                        (device_batch_size, 5, 5, 6, 4),
                        (device_batch_size, 3, 3, 4, 4),
                        (device_batch_size, 1, 1, 4, 4))]


def _tranposed_class_shapes(device_batch_size):
  return [jax.ShapedArray(shape, jnp.int32)
          for shape in ((device_batch_size, 38, 38, 4),
                        (device_batch_size, 19, 19, 6),
                        (device_batch_size, 10, 10, 6),
                        (device_batch_size, 5, 5, 6),
                        (device_batch_size, 3, 3, 4),
                        (device_batch_size, 1, 1, 4))]
# pylint: enable=g-complex-comprehension


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


def _run_cocoeval(all_core_detections, epoch, coco_times):
  """Computes eval score."""
  detections = []
  for predictions in all_core_detections:
    # There is an all gather operation at the end of eval.
    # Simply take the first core's data.
    predictions['detections'] = predictions['detections'][0]
    if ssd_constants.IS_PADDED in predictions:
      predictions[ssd_constants.IS_PADDED] = (
          predictions[ssd_constants.IS_PADDED][0])
    detections.append(merge_detections_on_a_single_host(predictions))

  coco_begin = time.time()
  accuracy = coco_metric.compute_map(
      detections, coco_gt, use_cpp_extension=True, nms_on_tpu=True)
  coco_times[epoch + 1] = time.time() - coco_begin
  mlp_log.mlperf_print(
      'eval_accuracy', accuracy['COCO/AP'], metadata={'epoch_num': epoch + 1})
  mlp_log.mlperf_print('eval_stop', None, metadata={'epoch_num': epoch + 1})
  if accuracy['COCO/AP'] > ssd_constants.EVAL_TARGET:
    mlp_log.mlperf_print(
        'run_stop', None, metadata={'status': 'success'})


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
                      enable_python_tracer=False,
                      host_trace_level=2)
  time.sleep(profile_time_sec)
  xprof_url = xprof.end_session_and_get_url(tag='')
  logging.info('Xprof profile is at %s', xprof_url)


def profile_with_xprof_on_background(start_after_sec=30, profile_time_sec=1,
                                     device='REDACTED'):
  profiler_thread = thread.ThreadPoolExecutor(jax.local_device_count(), 'xprof')
  profiler_thread.submit(partial(xprof_profile, start_after_sec,
                                 profile_time_sec, device))


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  Args:
    params: a parameter dictionary that includes lr_warmup_epoch,
      first_lr_drop_epoch, and second_lr_drop_epoch.
  """
  # Do not use steps_per_epoch from params, as we need a float value here.
  steps_per_epoch = params['num_examples_per_epoch'] / params['batch_size']
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(params['first_lr_drop_epoch'] *
                                     steps_per_epoch)
  params['second_lr_drop_step'] = int(params['second_lr_drop_epoch'] *
                                      steps_per_epoch)


def learning_rate_schedule(params, global_step):
  """Handles learning rate scaling, linear warmup, and learning rate decay.

  Args:
    params: A dictionary that defines hyperparameters of model.
    global_step: A tensor representing current global step.

  Returns:
    A tensor representing current learning rate.
  """
  base_learning_rate = params['base_learning_rate']
  lr_warmup_step = params['lr_warmup_step']
  first_lr_drop_step = params['first_lr_drop_step']
  second_lr_drop_step = params['second_lr_drop_step']
  batch_size = params['batch_size']
  scaling_factor = batch_size / ssd_constants.DEFAULT_BATCH_SIZE
  adjusted_learning_rate = base_learning_rate * scaling_factor
  learning_rate = (jnp.array(global_step).astype(jnp.float32) /
                   lr_warmup_step) * adjusted_learning_rate
  lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  if jax.host_id() == 0:
    mlp_log.mlperf_print('opt_learning_rate_warmup_factor',
                         adjusted_learning_rate / lr_warmup_step)
    mlp_log.mlperf_print('opt_learning_rate_warmup_steps', lr_warmup_step)
  for mult, start_global_step in lr_schedule:
    learning_rate = lax.cond(global_step < start_global_step, learning_rate,
                             lambda x: x, mult,
                             lambda mult: adjusted_learning_rate * mult)
  return learning_rate


def construct_run_config():
  """Construct the run config parameters.

  Returns:
    A dictionary containing run parameters.
  """

  global_batch_size = FLAGS.global_batch_size
  num_shards = jax.local_device_count() * jax.host_count()
  num_replicas = num_shards // FLAGS.num_partitions
  # Do not transpose input if spatial partitioning is enabled for now.
  transpose_input = False if FLAGS.num_partitions > 1 else FLAGS.transpose_input
  dtype = jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32
  return dict(
      base_learning_rate=FLAGS.base_learning_rate,
      batch_size=global_batch_size,  # global batch size
      host_batch_size=global_batch_size // jax.host_count(),  # Used in input_fn
      device_batch_size=global_batch_size // num_replicas,  # Used in model_fn
      eval_batch_size=FLAGS.eval_batch_size,  # global batch size
      host_eval_batch_size=FLAGS.eval_batch_size // jax.host_count(),
      device_eval_batch_size=FLAGS.eval_batch_size // num_replicas,
      conv0_space_to_depth=FLAGS.conv0_space_to_depth,
      dataset_index=jax.host_id(),
      dataset_num_shards=jax.host_count(),
      dbn_tile_col=-1,  # number of cols in each distributed batch norm group.
      dbn_tile_row=-1,  # number of rows in each distributed batch norm group.
      enable_wus=FLAGS.enable_wus,
      eval_every_checkpoint=False,
      eval_samples=FLAGS.eval_samples,
      first_lr_drop_epoch=FLAGS.first_lr_drop_epoch,
      second_lr_drop_epoch=FLAGS.second_lr_drop_epoch,
      iterations_per_loop=FLAGS.iterations_per_loop,
      local_device_count=jax.local_device_count(),
      lr_warmup_epoch=FLAGS.lr_warmup_epoch,
      model_dir=FLAGS.model_dir,
      num_epochs=FLAGS.num_epochs,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      num_shards=num_shards,
      num_replicas=num_replicas,
      local_num_replicas=jax.local_device_count() // FLAGS.num_partitions,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      steps_per_epoch=int(
          math.ceil(FLAGS.num_examples_per_epoch / global_batch_size)),
      tpu_slice_col=-1,
      tpu_slice_row=-1,
      transpose_input=transpose_input,
      use_bfloat16=FLAGS.use_bfloat16,
      dtype=dtype,
      bn_group_size=FLAGS.bn_group_size,
      use_spatial_partitioning=FLAGS.num_partitions > 1,
      num_partitions=FLAGS.num_partitions,
      val_json_file=FLAGS.val_json_file,
      visualize_dataloader=False,
      eval_steps=int(math.ceil(FLAGS.eval_samples / FLAGS.eval_batch_size)),
      weight_decay=ssd_constants.WEIGHT_DECAY,
  )


@partial(jax.jit, static_argnums=(1, 2, 3))
def create_model(key, device_batch_size, image_size, parameters):
  """Creates the SSD model architecture.

  Args:
    key: Int seed value for random number generators.
    device_batch_size: Int value for the local (device) batch size.
    image_size: Int value for lenght/width of the images.
    parameters : Dictionary holding model parameters.

  Returns:
      A pair consisting of an instance of Model and model state (nn.Collection).
  """
  if parameters['conv0_space_to_depth']:
    space_to_depth_bs = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    input_shape = (device_batch_size,
                   image_size // space_to_depth_bs,
                   image_size // space_to_depth_bs,
                   3 * (space_to_depth_bs**2))
  else:
    input_shape = (device_batch_size, image_size, image_size, 3)
  model_def = ssd_architecture.SSDModel.partial(parameters=parameters,
                                                axis_name='batch')
  with nn.stateful() as init_state:
    _, model = model_def.create_by_shape(key, [(input_shape,
                                                parameters['dtype'])])
  return model, init_state


def train_step(optimizer, state, batch, params):
  """Single training iteration.

  Args:
    optimizer: An instance of flax.optom.Optimizer.
    state: An instance of flax.nn.Collection.
    batch: Input batch for the iteration.
    params: A dictionary of run config parameters.

  Returns:
    An instance of the updated flax.optom.Optimizer and updated state.
  """
  (features, labels) = batch
  train_pad_size = input_pipeline.get_spmd_image_padding_size(
      params, features.shape[1:])
  if train_pad_size:
    features = features[:, :-train_pad_size, :, :]

  if params['transpose_input']:
    if params['device_batch_size'] > 8:
      features = jnp.transpose(features, [3, 0, 1, 2])
    else:
      features = jnp.transpose(features, [2, 0, 1, 3])
    labels[ssd_constants.BOXES] = jnp.transpose(labels[ssd_constants.BOXES],
                                                [2, 0, 1])
  scale_size = 1
  if params['conv0_space_to_depth']:
    space_to_depth_bs = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    # Depth increased by block_size * block_size from width and height
    # in space to depth transform in conv0.
    scale_size = space_to_depth_bs**2

    features -= jnp.reshape(
        jnp.array(list(ssd_constants.NORMALIZATION_MEAN) * scale_size,
                  dtype=features.dtype),
        [1, 1, 3 * scale_size])

    features /= jnp.reshape(
        jnp.array(list(ssd_constants.NORMALIZATION_STD) * scale_size,
                  dtype=features.dtype),
        [1, 1, 3 * scale_size])

  def loss_fn(model):
    """SSD model loss function.

    Args:
      model: An instance of flax.optom.Optimizer.target.

    Returns:
      Loss and new state.
    """
    with nn.stateful(state) as new_state:
      class_outputs, box_outputs = model(features)

    levels = class_outputs.keys()
    for level in levels:
      class_outputs[level] = class_outputs[level].astype(jnp.float32)
      box_outputs[level] = box_outputs[level].astype(jnp.float32)

    loss, _, _ = ssd_model.detection_loss(class_outputs, box_outputs, labels)
    return loss, new_state

  lr = learning_rate_schedule(params, optimizer.state.step)
  new_optimizer, _, new_state = optimizer.optimize(loss_fn, learning_rate=lr)
  return new_optimizer, new_state


def eval_step(model, state, batch, device_id, params):
  """Single eval iteration.

  Args:
    model: Forward pass model e.g., instance of Optimizer.Target.
    state: An instance of flax.nn.Collection.
    batch: Input batch for the iteration.
    device_id: Int value denoting the core index to be used in all gather.
    params: Parameter dictionary.

  Returns:
    A dictionary that contains two fields.
      'predictions': the result of the predictions.
      'padded': Optional field, correspond to a tensor with bools denoting
      whether the examples were padded or not.
  """
  (features, labels) = batch
  train_pad_size = input_pipeline.get_spmd_image_padding_size(
      params, features.shape[1:])
  if train_pad_size:
    features = features[:, :-train_pad_size, :, :]

  scale_size = 1
  if params['conv0_space_to_depth']:
    space_to_depth_bs = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    # Depth increased by block_size * block_size from width and height
    # in space to depth transform in conv0.
    scale_size = space_to_depth_bs**2

    features -= jnp.reshape(
        jnp.array(list(ssd_constants.NORMALIZATION_MEAN) * scale_size,
                  dtype=features.dtype),
        [1, 1, 3 * scale_size])

    features /= jnp.reshape(
        jnp.array(list(ssd_constants.NORMALIZATION_STD) * scale_size,
                  dtype=features.dtype),
        [1, 1, 3 * scale_size])

  with nn.stateful(state, mutable=False):
    class_outputs, box_outputs = model(features, train=False)

  levels = class_outputs.keys()
  for level in levels:
    class_outputs[level] = class_outputs[level].astype(jnp.float32)
    box_outputs[level] = box_outputs[level].astype(jnp.float32)

  flattened_cls, flattened_box = ssd_model.concat_outputs(class_outputs,
                                                          box_outputs)

  # TODO(deveci): There are a lot of computations in box_decode that does not
  # depend on flattened_box input. Make sure that such computations do not
  # happen in every iteration.
  decoded_boxes = ssd_model.box_decode(flattened_box)

  pred_scores = jax.nn.softmax(flattened_cls, axis=2)

  pred_scores, indices = ssd_model.select_top_k_scores(
      pred_scores, ssd_constants.MAX_NUM_EVAL_BOXES)

  detections = ssd_model.non_max_suppression(
      scores_in=pred_scores,
      boxes_in=decoded_boxes,
      top_k_indices=indices,
      labels=labels)
  predictions = dict(detections=detections)
  if ssd_constants.IS_PADDED in labels:
    predictions[ssd_constants.IS_PADDED] = labels[ssd_constants.IS_PADDED]
  predictions = all_gather_detections(device_id, predictions)
  return predictions


def merge_detections_on_a_single_host(predictions):
  """Merges the predictions from different cores on a single host using numpy functions.

  Args:
    predictions: A dictionary that contains two fields.
      'predictions': the result of the predictions.
      'padded': Optional field, correspond to a tensor with bools denoting
      whether the examples were padded or not.
  Returns:
    A filtered list from prediction['detections'].
  """
  num_detection_result_count = 7
  predictions['detections'] = jax.device_get(predictions['detections'])

  # If there is only a single host, work on the merged data in the host.
  predictions['detections'] = onp.reshape(
      predictions['detections'],
      [-1, ssd_constants.MAX_NUM_EVAL_BOXES, num_detection_result_count])
  if ssd_constants.IS_PADDED in predictions:
    predictions[ssd_constants.IS_PADDED] = jax.device_get(
        predictions[ssd_constants.IS_PADDED])
    predictions[ssd_constants.IS_PADDED] = onp.reshape(
        predictions[ssd_constants.IS_PADDED], [-1])

    predictions['detections'] = predictions['detections'][
        onp.logical_not(predictions[ssd_constants.IS_PADDED])]
  return predictions['detections']


@jax.jit
def all_gather_detections(device_id, predictions):
  """Gathers the predictions from different cores on all cores on using psum.

  Args:
    device_id: global device index.
    predictions: A dictionary that contains two fields.
      'predictions': the result of the predictions.
      'padded': Optional field, correspond to a tensor with bools denoting
      whether the examples were padded or not.
  Returns:
    A filtered list from prediction['detections'].
  """

  eval_batch_size = FLAGS.eval_batch_size
  num_detection_result_count = 7
  num_replicas = (jax.device_count() // FLAGS.num_partitions)

  # Create a new detection array with extra dimension of size num_replicas
  global_detections = jnp.zeros(
      [num_replicas, eval_batch_size // num_replicas,
       ssd_constants.MAX_NUM_EVAL_BOXES, num_detection_result_count]
      ).astype(jnp.float32)
  # Each TPU core places their own data to core-id = device_id in the global
  # array.
  # predictions['detections'] contains data from local_cores so access it
  # with local core id = device_id % jax.local_device_count()
  global_detections = jax.ops.index_update(
      global_detections, device_id, predictions['detections'])
  # Perform an all reduce to gather data in all cores.
  global_detections = jax.lax.psum(global_detections, axis_name='batch')
  predictions['detections'] = global_detections

  if ssd_constants.IS_PADDED in predictions:
    # If padding information is provided also gather all padding info.
    global_padding = jnp.zeros([num_replicas, eval_batch_size // num_replicas])
    global_padding = jax.ops.index_update(
        global_padding, device_id,
        predictions[ssd_constants.IS_PADDED].astype(jnp.int32))
    global_padding = jax.lax.psum(global_padding, axis_name='batch')
    predictions[ssd_constants.IS_PADDED] = global_padding
  return predictions


@jax.jit
def local_barrier_helper():
  return jax.pmap(lambda x: x)(jnp.ones(jax.local_device_count()))


def local_barrier():
  local_barrier_helper().block_until_ready()


@jax.jit
def global_barrier_helper():
  d = jax.random.normal(jax.random.PRNGKey(0), [jax.local_device_count()])
  val = jax.pmap(lambda x: jax.lax.psum(x, axis_name='i'), axis_name='i')(d)
  return val


def global_barrier():
  val = global_barrier_helper()
  local_barrier()
  return val


def get_device_assignment():
  """Get device assignemnt permutation."""
  if FLAGS.num_partitions == 1:
    return jax.devices(), [[d] for d in jax.local_devices()]

  def bounds_from_last_device(device):
    x, y, _ = device.coords
    return (x + 1), (y + 1)

  jax_devices = jax.devices()
  host_count = jax.host_count()
  # TODO(deveci): get per host bounds from JAX backend.
  per_host_x, per_host_y = bounds_from_last_device(jax.local_devices(0)[-1])
  device_map = onp.ndarray(
      shape=(host_count, per_host_x, per_host_y, 2), dtype=int)

  for core_id, device in enumerate(jax_devices):
    host_id = device.host_id
    x, y, _ = device.coords
    core_on_chip = device.core_on_chip
    device_map[host_id][x % per_host_x][y % per_host_y][core_on_chip] = core_id

  num_partitions = FLAGS.num_partitions
  replicas_per_host = jax.local_device_count() // num_partitions
  inner_y = min(num_partitions // 2, 2)
  inner_x = (num_partitions // 2) // inner_y
  # Set inner_ring within each replica.
  permute = list(range(num_partitions // 2)) + list(
      range(num_partitions - 1, num_partitions // 2 - 1, -1))

  device_assignment = []
  local_device_assignment = []
  for host_id in range(host_count):
    for replica_id in range(replicas_per_host):
      x_start = replica_id * inner_x
      cores_in_replica = []
      for y in range(inner_y):
        for x in range(x_start, x_start + inner_x):
          for core_on_chip in range(2):
            core_id = device_map[host_id][x][y][core_on_chip]
            cores_in_replica.append(core_id)
      # Set inner_ring within each replica for better performance.
      cores_in_replica = [cores_in_replica[i] for i in permute]
      replica_devices = [jax_devices[i] for i in cores_in_replica]
      if host_id == jax.host_id():
        local_device_assignment.append(replica_devices)
      device_assignment.extend(replica_devices)

  return device_assignment, local_device_assignment


def per_host_sum(x):
  return jax.lax.psum(x, 'hosts')


def per_host_sum_pmap(in_tree):
  """Execute sum on in_tree's leaves over ICI."""
  ldc = jax.local_device_count()
  host_psum = jax.pmap(per_host_sum, axis_name='hosts')
  def pre_pmap(x):
    y = onp.zeros((ldc, *x.shape), dtype=x.dtype)
    y[0] = x
    return y
  def post_pmap(x):
    return jax.device_get(x)[0]
  return jax.tree_map(post_pmap, host_psum(jax.tree_map(pre_pmap, in_tree)))


def main(argv):
  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL

  del argv

  tf.enable_v2_behavior()
  for _ in range(FLAGS.repeat_experiment):
    run_ssd()


def run_ssd():
  """Runs a single end to end ssd experiment."""
  params = construct_run_config()
  host_id = jax.host_id()
  if FLAGS.infeed:
    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')
  if FLAGS.seed is not None:
    seed = FLAGS.seed
  else:
    seed = onp.uint32(time.time() if jax.host_id() == 0 else 0)
    seed = per_host_sum_pmap(seed)

  rng = random.PRNGKey(seed)
  tf.random.set_seed(seed)
  device_assignment, local_device_assignment = get_device_assignment()

  # TODO(deveci): check if replicate_devices = jax.local_devices() works.
  replicate_devices = [assignment[0] for assignment in local_device_assignment]

  replicate_fn = partial(jax_utils.replicate, devices=replicate_devices)
  coco_thread = None
  if host_id == 0:
    mlp_log.mlperf_print(key='cache_clear', value=True)
    mlp_log.mlperf_print(key='init_start', value=None)
    mlp_log.mlperf_print('global_batch_size', params['batch_size'])
    mlp_log.mlperf_print('opt_base_learning_rate', params['host_batch_size'])
    mlp_log.mlperf_print('opt_weight_decay', params['weight_decay'])
    mlp_log.mlperf_print(
        'model_bn_span', params['batch_size'] // params['num_shards'] *
        params['bn_group_size'])
    mlp_log.mlperf_print(
        'opt_learning_rate_decay_boundary_epochs',
        [params['first_lr_drop_epoch'], params['second_lr_drop_epoch']])
    mlp_log.mlperf_print('max_samples', ssd_constants.NUM_CROP_PASSES)
    mlp_log.mlperf_print('train_samples', FLAGS.num_examples_per_epoch)
    mlp_log.mlperf_print('eval_samples', FLAGS.eval_samples)
    mlp_log.mlperf_print('seed', int(seed))

  coco_thread = thread.ThreadPoolExecutor(8, 'coco_thread')
  model_dtype = params['dtype']
  update_learning_rate_schedule_parameters(params)
  model, state = create_model(
      rng,
      params['device_batch_size'],
      ssd_constants.IMAGE_SIZE,
      params)
  optimizer = optim.Momentum(
      learning_rate=params['base_learning_rate'],
      beta=ssd_constants.MOMENTUM,
      weight_decay=params['weight_decay']).create(model)

  if params['resnet_checkpoint']:
    optimizer, state = ssd_resnet_checkpoint_reader.load_from_tf_checkpoints(
        optimizer, state, params)

  state = replicate_fn(state)
  optimizer = optimizer.replicate(devices=replicate_devices)

  del model  # do not keep a copy of the initial model

  if params['transpose_input']:
    # Transpose input should be false when spmd is used.
    assert FLAGS.num_partitions == 1
    image_partition = sharded_jit.PartitionSpec(FLAGS.num_partitions, 1, 1, 1)
  else:
    image_partition = sharded_jit.PartitionSpec(1, FLAGS.num_partitions, 1, 1)
  # Eval does not apply double transpose trick.
  eval_image_partition = sharded_jit.PartitionSpec(1, FLAGS.num_partitions, 1,
                                                   1)

  eval_partitions = ((None, None, (eval_image_partition, None), None), None)
  train_partitions = ((None, None, (image_partition, None)), None)
  s_train_step = partial(train_step, params=params)
  if FLAGS.num_partitions > 1:
    s_train_step = sharded_jit.sharded_jit(s_train_step,
                                           in_parts=train_partitions[0],
                                           out_parts=train_partitions[1])

  p_train_step = jax.pmap(s_train_step,
                          axis_name='batch', axis_size=params['num_replicas'])
  s_eval_step = partial(eval_step, params=params)
  if FLAGS.num_partitions > 1:
    s_eval_step = sharded_jit.sharded_jit(s_eval_step,
                                          in_parts=eval_partitions[0],
                                          out_parts=eval_partitions[1])

  p_eval_step = jax.pmap(s_eval_step, axis_name='batch',
                         axis_size=params['num_replicas'])

  host_step = 0

  train_ds = input_pipeline.ssd_input_pipeline(
      params,
      FLAGS.training_file_pattern,
      is_training=True,
      use_fake_data=FLAGS.use_fake_data,
      host_batch_size=params['host_batch_size'],
      transpose_input=params['transpose_input'])

  eval_steps = params['eval_steps']
  eval_ds = input_pipeline.ssd_input_pipeline(
      params,
      FLAGS.validation_file_pattern,
      is_training=False,
      use_fake_data=FLAGS.use_fake_data,
      distributed_eval=True,
      count=eval_steps * FLAGS.eval_batch_size,
      host_batch_size=params['host_eval_batch_size'])

  flat_device_id = jnp.arange(
      params['local_num_replicas'] * host_id,
      params['local_num_replicas'] * (1 + host_id))

  eval_epochs = ssd_constants.EVAL_EPOCHS
  if host_id == 0:
    logging.info('eval_epochs %s', str(eval_epochs))
  # Set the input shapes

  num_corners = 4
  train_batch_dims = (params['local_num_replicas'],
                      params['device_batch_size'])

  space_to_depth_bs = 1
  if params['conv0_space_to_depth']:
    space_to_depth_bs = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    image_shape = (ssd_constants.IMAGE_SIZE // space_to_depth_bs,
                   ssd_constants.IMAGE_SIZE // space_to_depth_bs,
                   3 * (space_to_depth_bs ** 2))
  else:
    image_shape = (ssd_constants.IMAGE_SIZE, ssd_constants.IMAGE_SIZE, 3)

  train_pad_size = input_pipeline.get_spmd_image_padding_size(params,
                                                              image_shape)
  if train_pad_size:
    image_padding = (train_pad_size, 0, 0)
    image_shape = tuple([sum(x) for x in zip(image_shape, image_padding)])
  # If tranpose_input is true, train images and boxes will have different
  # shape.
  if params['transpose_input']:
    if params['device_batch_size'] > 8:
      train_input_shape = ((params['local_num_replicas'],) + image_shape +
                           (params['device_batch_size'],))
    else:
      train_input_shape = (params['local_num_replicas'],
                           ssd_constants.IMAGE_SIZE // space_to_depth_bs,
                           ssd_constants.IMAGE_SIZE // space_to_depth_bs,
                           params['device_batch_size'],
                           3 * (space_to_depth_bs**2))

    train_box_shape = (params['local_num_replicas'],
                       ssd_constants.NUM_SSD_BOXES, num_corners,
                       params['device_batch_size'])
  else:
    train_input_shape = train_batch_dims + image_shape
    train_box_shape = train_batch_dims + (ssd_constants.NUM_SSD_BOXES,
                                          num_corners)

  train_class_shape = train_batch_dims + (ssd_constants.NUM_SSD_BOXES,)

  eval_batch_dims = (params['local_num_replicas'],
                     params['device_eval_batch_size'])
  eval_input_shape = eval_batch_dims + image_shape
  eval_box_shape = eval_batch_dims + (ssd_constants.MAX_NUM_EVAL_BOXES,
                                      num_corners)
  eval_class_shape = eval_batch_dims + (ssd_constants.MAX_NUM_EVAL_BOXES,)
  eval_row_shape_shape = eval_batch_dims + (3,)

  # Done with setting the input shapes.
  def device_train_loop_cond(args):
    _, _, _, step, epoch = args
    return step // params['steps_per_epoch'] == epoch
  def device_train_loop_body(args):
    """Device training loop body."""
    optimizer, state, token, step, epoch = args
    if not params['use_spatial_partitioning']:
      partitions = (image_partition,
                    None,  # boxes
                    None,  # classes
                    None,  # num_classes
                    )

      (images, boxes, classes, num_unmatched_boxes), token = lax.infeed(
          token, shape=(
              jax.ShapedArray(train_input_shape[1:], model_dtype),
              jax.ShapedArray(train_box_shape[1:], jnp.float32),
              jax.ShapedArray(train_class_shape[1:], jnp.int32),
              jax.ShapedArray(train_batch_dims[1:], jnp.float32)),
          partitions=partitions)
    else:
      partitions = (image_partition,
                    None, None, None, None, None, None,  # boxes
                    None, None, None, None, None, None,  # classes
                    None,  # num_classes
                    )
      box_shapes = _tranposed_box_shapes(params['device_batch_size'])
      class_shapes = _tranposed_class_shapes(params['device_batch_size'])
      (images,
       boxes0, boxes1, boxes2, boxes3, boxes4, boxes5,
       classes0, classes1, classes2, classes3, classes4, classes5,
       num_unmatched_boxes), token = lax.infeed(
           token,
           shape=(jax.ShapedArray(train_input_shape[1:], model_dtype),
                  box_shapes[0], box_shapes[1], box_shapes[2],
                  box_shapes[3], box_shapes[4], box_shapes[5],
                  class_shapes[0], class_shapes[1], class_shapes[2],
                  class_shapes[3], class_shapes[4], class_shapes[5],
                  jax.ShapedArray(train_batch_dims[1:], jnp.float32)),
           partitions=partitions)
      boxes = {0: boxes0, 1: boxes1, 2: boxes2, 3: boxes3, 4: boxes4, 5: boxes5}
      classes = {
          0: classes0,
          1: classes1,
          2: classes2,
          3: classes3,
          4: classes4,
          5: classes5
      }
    batch = (images, {ssd_constants.BOXES: boxes,
                      ssd_constants.CLASSES: classes,
                      ssd_constants.NUM_MATCHED_BOXES: num_unmatched_boxes})
    optimizer, state = train_step(optimizer, state, batch, params)
    step += 1
    return optimizer, state, token, step, epoch
  def device_train_loop(optimizer, state, step, epoch):
    """Device training loop."""
    token = lax.create_token(step)
    optimizer, state, _, step, _ = lax.while_loop(
        device_train_loop_cond,
        device_train_loop_body,
        (optimizer, state, token, step, epoch))
    state = lax.pmean(state, 'batch')
    return optimizer, state, step

  if FLAGS.num_partitions > 1:
    s_device_train_loop = sharded_jit.sharded_jit(
        device_train_loop, in_parts=None, out_parts=None)
  else:
    s_device_train_loop = device_train_loop

  if FLAGS.enable_wus:
    p_train_epoch = jax.pmap(
        s_device_train_loop,
        axis_name='batch',
        in_axes=(None, 0, None, None),
        axis_size=params['num_replicas'],
        devices=device_assignment)
  else:
    p_train_epoch = jax.pmap(
        s_device_train_loop,
        axis_name='batch',
        axis_size=params['num_replicas'],
        devices=device_assignment)

  # TODO(deveci): Currently mixing this with spmd is failing.
  # re-enable this once figured out.
  # reduce_state_fn = jax.pmap(
  #    lambda x: jax.lax.pmean(x, 'batch'),
  #    axis_name='batch',
  #    axis_size=params['num_replicas'])
  if FLAGS.precompile:
    if FLAGS.infeed:
      if FLAGS.enable_wus:
        # the device training loop condition will immediately be false
        p_train_epoch(unbroadcast(optimizer), state, 0, 1)
      else:
        # the device training loop condition will immediately be false
        p_train_epoch(optimizer, state, replicate_fn(0), replicate_fn(1))
    else:
      batch = (
          jnp.zeros(train_input_shape, model_dtype),  # image
          {
              ssd_constants.BOXES:
                  jnp.zeros(train_box_shape, jnp.float32),
              ssd_constants.CLASSES:
                  jnp.zeros(train_class_shape, jnp.int32),
              ssd_constants.NUM_MATCHED_BOXES:
                  jnp.zeros(train_batch_dims, jnp.float32)
          }  # train labels
      )
      # Dummy train step to force it to compile.
      p_train_step(optimizer, state, batch)
    eval_labels = {
        ssd_constants.BOXES: jnp.zeros(eval_box_shape, jnp.float32),
        ssd_constants.CLASSES: jnp.zeros(eval_class_shape, jnp.float32),
        ssd_constants.SOURCE_ID: jnp.zeros(eval_batch_dims, jnp.int32),
        ssd_constants.RAW_SHAPE: jnp.zeros(eval_row_shape_shape, jnp.int32)}

    # If the eval is steps require more input than the existing samples,
    # then the labels also have a field ssd_constants.IS_PADDED
    if eval_steps * FLAGS.eval_batch_size > params['eval_samples']:
      eval_labels[ssd_constants.IS_PADDED] = jnp.zeros(eval_batch_dims,
                                                       jnp.int32).astype('bool')
      logging.info('Precompiling eval with IS_PADDED labels.')
    else:
      logging.info('Precompiling eval without IS_PADDED labels.')
    batch = (jnp.zeros(eval_input_shape, model_dtype), eval_labels)
    detections = p_eval_step(optimizer.target, state, batch, flat_device_id)
    # TODO(deveci): Currently mixing this with spmd is failing.
    # state = reduce_state_fn(state)

  if host_id == 0:
    if FLAGS.profile:
      profile_with_xprof_on_background()

  train_iter = iter(train_ds)
  eval_iter = iter(eval_ds)

  # access data after run_start
  def create_coco_gt():
    global coco_gt
    coco_gt = coco_metric.create_coco(
        FLAGS.val_json_file, use_cpp_extension=True)

  create_coco_gt_thread = thread.ThreadPoolExecutor(1, 'create_coco_gt')
  time.sleep(20)
  global_barrier()

  if host_id == 0:
    mlp_log.mlperf_print('init_stop', None)
    mlp_log.mlperf_print('run_start', None)
  create_coco_gt_thread.submit(create_coco_gt)

  host_step, device_step = 0, replicate_fn(0)

  train_times = {}
  eval_times = {}
  coco_times = {}
  ds_times, core_eval_times, host_merge_times = {}, {}, {}
  for epoch in range(params['num_epochs']):
    if host_id == 0:
      mlp_log.mlperf_print(
          'block_start',
          None,
          metadata={
              'first_epoch_num': epoch + 1,
              'epoch_count': 1
          })
    if FLAGS.detailed_time:
      train_begin = time.time()
    if FLAGS.infeed:
      if FLAGS.enable_wus:
        optimizer, state, device_step = p_train_epoch(
            unbroadcast(optimizer), state,
            host_step, epoch)
      else:
        optimizer, state, device_step = p_train_epoch(
            optimizer, state, device_step, replicate_fn(epoch))
    while int(host_step // params['steps_per_epoch']) == epoch:
      # pylint: disable=protected-access
      # As JAX uses Tensorflow datasets, we need to extract the raw numpy
      # array from the eager tf.tensor
      batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))
      # pylint: enable=protected-access

      if FLAGS.infeed:
        (images, labels) = batch
        boxes = labels[ssd_constants.BOXES]
        classes = labels[ssd_constants.CLASSES]
        num_unmatched_boxes = labels[ssd_constants.NUM_MATCHED_BOXES]

        assert (images.shape == train_input_shape and
                images.dtype == model_dtype)
        if params['use_spatial_partitioning']:
          # Boxes and classes are dictionaries when spatially partitioned.
          # Each of them have keys from 0 to 6.
          # The sizes for boxes are
          # 0 : (local_num_replicas, device_batch_size,
          #   38, 38, 4, 4),
          #  1: (local_num_replicas, device_batch_size,
          #      19, 19, 6, 4),
          #  2: (local_num_replicas, device_batch_size,
          #      10, 10, 6, 4),
          #  3: (local_num_replicas, device_batch_size,
          #      5, 5, 6, 4),
          #  4: (local_num_replicas, device_batch_size,
          #      3, 3, 4, 4),
          #  5: (local_num_replicas, device_batch_size,
          #      1, 1, 4, 4)
          # Below we convert this dictionary into a new dictionary.
          # boxes is a dictionary with keys range(local_num_replicas)
          # Each boxes[replica_id] holds a tuple of size 6,
          # ((device_batch_size, 38, 38, 4, 4),
          #  (device_batch_size, 19, 19, 6, 4),
          #  (device_batch_size, 10, 10, 6, 4),
          #  (device_batch_size, 5, 5, 6, 4),
          #  (device_batch_size, 3, 3, 4, 4),
          #  (device_batch_size, 1, 1, 4, 4))

          # Similarly, classes have
          # expected_class_shapes = {
          # 0: (local_num_replicas, device_batch_size, 38, 38, 4),
          # 1: (local_num_replicas, device_batch_size, 19, 19, 6),
          # 2: (local_num_replicas, device_batch_size, 10, 10, 6),
          # 3: (local_num_replicas, device_batch_size, 5, 5, 6),
          # 4: (local_num_replicas, device_batch_size, 3, 3, 4),
          # 5: (local_num_replicas, device_batch_size, 1, 1, 4) }
          # After the conversion new_classes[replica_id] will hold a tuple of
          # size 6 with shapes:
          # Each boxes[replica_id] holds a tuple of size 6,
          # ((device_batch_size, 38, 38, 4),
          #  (device_batch_size, 19, 19, 6),
          #  (device_batch_size, 10, 10, 6),
          #  (device_batch_size, 5, 5, 6),
          #  (device_batch_size, 3, 3, 4),
          #  (device_batch_size, 1, 1, 4))
          new_boxes = {}
          new_classes = {}
          for i in range(params['local_num_replicas']):
            new_boxes[i] = [boxes[j][i] for j in range(6)]
            new_classes[i] = [classes[j][i] for j in range(6)]
          boxes = new_boxes
          classes = new_classes
        else:
          assert (boxes.shape == train_box_shape and
                  boxes.dtype == jnp.float32)
          assert (classes.shape == train_class_shape and
                  classes.dtype == jnp.int32)
        assert (num_unmatched_boxes.shape == train_batch_dims and
                num_unmatched_boxes.dtype == jnp.float32)
        for i in range(params['local_num_replicas']):
          replica_image = images[i]
          if params['transpose_input']:
            partition_axis = 0
            axis_size = replica_image.shape[partition_axis]
            chunk_size = axis_size // FLAGS.num_partitions

            replica_image_shards = [
                replica_image[i:i + chunk_size]
                for i in range(0, axis_size, chunk_size)
            ]
          else:
            partition_axis = 1
            axis_size = replica_image.shape[partition_axis]
            chunk_size = axis_size // FLAGS.num_partitions

            replica_image_shards = [
                replica_image[:, i:i + chunk_size]
                for i in range(0, axis_size, chunk_size)
            ]

          replica_devices = local_device_assignment[i]

          for img_shard, device in zip(replica_image_shards, replica_devices):
            if params['use_spatial_partitioning']:
              infeed_pool.submit(partial(
                  device.transfer_to_infeed,
                  (img_shard,
                   boxes[i][0], boxes[i][1], boxes[i][2],
                   boxes[i][3], boxes[i][4], boxes[i][5],
                   classes[i][0], classes[i][1], classes[i][2],
                   classes[i][3], classes[i][4], classes[i][5],
                   num_unmatched_boxes[i])))
            else:
              infeed_pool.submit(partial(
                  device.transfer_to_infeed,
                  (images[i], boxes[i], classes[i], num_unmatched_boxes[i])))
      else:
        optimizer, state = p_train_step(optimizer, state, batch)

      host_step += 1
    if FLAGS.detailed_time:
      local_barrier()
      train_times[epoch] = time.time() - train_begin
    if host_id == 0:
      mlp_log.mlperf_print(
          'block_stop',
          None,
          metadata={
              'first_epoch_num': epoch + 1,
              'epoch_count': 1
          })
    if epoch + 1 not in eval_epochs:
      continue
    if host_id == 0:
      mlp_log.mlperf_print(
          'eval_start', None, metadata={'epoch_num': epoch + 1})
    # TODO(deveci): Currently mixing this with spmd is failing.
    # state = reduce_state_fn(state)
    if FLAGS.detailed_time:
      eval_begin = time.time()
      ds_time, core_eval_time, host_merge_time = 0, 0, 0
    detections = []

    for _ in range(eval_steps):
      if FLAGS.detailed_time:
        ds_begin = time.time()

      # pylint: disable=protected-access
      # As JAX uses Tensorflow datasets, we need to extract the raw numpy
      # array from the eager tf.tensor
      batch = jax.tree_map(lambda x: x._numpy(), next(eval_iter))
      if FLAGS.detailed_time:
        core_eval_begin = time.time()
        ds_time += core_eval_begin - ds_begin

      # pylint: enable=protected-access
      predictions = p_eval_step(optimizer.target, state, batch, flat_device_id)
      detections.append(predictions)
      if FLAGS.detailed_time:
        local_barrier()
        merge_begin = time.time()
        core_eval_time += merge_begin - core_eval_begin
      if FLAGS.detailed_time:
        host_merge_time += time.time() - merge_begin
    if FLAGS.detailed_time:
      eval_times[epoch + 1] = time.time() - eval_begin
      core_eval_times[epoch + 1] = core_eval_time
      ds_times[epoch + 1] = ds_time
      host_merge_times[epoch + 1] = host_merge_time

    # distribute coco_eval on each host.
    epoch_id = eval_epochs.index(epoch + 1)
    epoch_id = epoch_id % jax.host_count()
    if host_id == epoch_id:
      coco_thread.submit(partial(_run_cocoeval, detections, epoch, coco_times))

  if host_id == 0:
    logging.info('train_time %f', sum(train_times.values()))
    logging.info('eval_time %f', sum(eval_times.values()))
    logging.info('coco_time %f', sum(coco_times.values()))

    logging.info('ds_times %f', sum(ds_times.values()))
    logging.info('core_eval_times %f', sum(core_eval_times.values()))
    logging.info('host_merge_times %f', sum(host_merge_times.values()))

    logging.info('train_times %s', str(train_times))
    logging.info('eval_times %s', str(eval_times))
    logging.info('coco_times %s', str(coco_times))

    logging.info('ds_times %s', str(ds_times))
    logging.info('core_eval_times %s', str(core_eval_times))
    logging.info('host_merge_times %s', str(host_merge_times))

  # Wait until computations are done before exiting
  global_barrier()


if __name__ == '__main__':
  app.run(main)
