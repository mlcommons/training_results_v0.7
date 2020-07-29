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
"""Utility functions for image models."""

import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.python.tpu.ops import tpu_ops
from REDACTED.tensorflow.python.training import moving_averages


def linear_2d(m, n):
  """Ring-order on a linearized mxn mesh.

  Here we go along the inner dimension of
  size n followed by the outer dimension of size m.

  Args:
    m: size of mesh outer dimension
    n: size of mesh inner dimension

  Returns:
    a list of mxn pairs
  """
  if m == 1:
    return [(0, i) for i in range(n)]
  if n == 1:
    return [(i, 0) for i in range(m)]
  if m % 2 != 0:
    return [(i % m, i // m) for i in range(n * m)]
  ret = []
  for i in range(m // 2):
    for j in range(n):
      ret.append((2 * i, j))
    for j in range(n - 1, -1, -1):
      ret.append((2 * i + 1, j))
  return ret


def snake_2d(m, n):
  """Snake ring-order on an mxn mesh.

  Here we snake around the mesh.
  This mapping is enables strict
  near neighbor communication on the 2-D mesh.

  Args:
    m: size of mesh outer dimension
    n: size of mesh inner dimension

  Returns:
    a list of mxn pairs
  """
  if m == 1:
    return [(0, i) for i in range(n)]
  if n == 1:
    return [(i, 0) for i in range(m)]
  if m % 2 != 0:
    return [(i % m, i // m) for i in range(n * m)]
  ret = [(0, 0)]
  for i in range(m // 2):
    for j in range(1, n):
      ret.append((2 * i, j))
    for j in range(n - 1, 0, -1):
      ret.append((2 * i + 1, j))
  for i in range(m - 1, 0, -1):
    ret.append((i, 0))
  return ret


def coord2core_id(num_cores_per_row, order_2d, ncores=2):
  """Linearize the two cores on a chip.

  Args:
    num_cores_per_row: Number of cores in the inner dimension
    order_2d: chip order along the mesh
    ncores: number of cores per chip

  Returns:
    a list of mxn pairs
  """

  output = []
  for index in order_2d:
    row, col = index
    for i in range(ncores):
      output.append(row * num_cores_per_row + col * ncores + i)
  return output


def tile_group_assignment(physical_shape, tile_shape, num_groups):
  """Create group assignment for TPU cores."""

  # Nmber of rows and cols of TPU chip topology. Each chip has 2 cores.
  _, physical_shape_col = physical_shape
  # Number of rows and columns in each TPU chip subgroup. Each chip has 2 cores.
  tile_shape_row, tile_shape_col = tile_shape
  first_group = coord2core_id(
      physical_shape_col * 2,
      linear_2d(tile_shape_row, tile_shape_col),
      ncores=2)

  group_assignment = []
  cores_per_row_in_tile = tile_shape_col * 2
  for i in range(num_groups):
    tiles_per_col = physical_shape_col // tile_shape_col
    offset = i // tiles_per_col * tiles_per_col * tile_shape_row + i % tiles_per_col
    new_group = [y + offset * cores_per_row_in_tile for y in first_group]
    group_assignment.append(new_group)
  return group_assignment


def sp_tile_group_assignment(physical_shape, tile_shape, num_groups):
  """Create group assignment for spatial partitioning."""
  # Number of rows and columns of the TPU replica topology.
  physical_shape_row, physical_shape_col = physical_shape
  tile_shape_row, tile_shape_col = tile_shape
  first_group = coord2core_id(
      physical_shape_col, snake_2d(tile_shape_row, tile_shape_col), ncores=1)

  group_assignment = []
  logical_to_physical = dict(
      zip(
          coord2core_id(
              physical_shape_col,
              snake_2d(physical_shape_row, physical_shape_col),
              ncores=1),
          [i for i in range(physical_shape_col * physical_shape_row)]))

  replicas_per_row_in_tile = tile_shape_col
  tiles_per_col = physical_shape_col // tile_shape_col
  for i in range(num_groups):
    offset = i // tiles_per_col * tiles_per_col * tile_shape_row + i % tiles_per_col
    new_group_logical = [
        y + offset * replicas_per_row_in_tile for y in first_group
    ]
    new_group_physical = [
        logical_to_physical[index] for index in new_group_logical
    ]
    group_assignment.append(new_group_physical)

  return group_assignment


def auto_select_shapes(num_shards, distributed_group_size,
                       input_partition_dims):
  """Auto-select shapes based on number of replicas and distributed group size.

  Args:
    num_shards: number of replicas
    distributed_group_size: number of replicas in the distributed batch norm.
    input_partition_dims: model parallelism dimensions

  Returns:
    physical_shape and tile_shape
  """

  physical_shape = None
  tile_shape = None
  model_cores = 1
  if input_partition_dims:
    model_cores = np.prod(input_partition_dims)
  num_cores = num_shards * model_cores
  # Compute physical shapes first.
  if num_cores == 8192:
    physical_shape = (32, 128)
  if num_cores == 4096:
    physical_shape = (32, 64)
  if num_cores == 2048:
    physical_shape = (32, 32)
  if num_cores == 1024:
    physical_shape = (32, 16)
  if num_cores == 512:
    physical_shape = (16, 16)
  if num_cores == 256:
    physical_shape = (16, 8)
  if num_cores == 128:
    physical_shape = (8, 8)

  input_size = (1, 1)
  if input_partition_dims:
    if model_cores == 4:
      input_size = (2, 1)
    if model_cores == 8:
      input_size = (2, 2)
  if physical_shape:
    physical_shape = tuple(np.array(physical_shape) // np.array(input_size))

  # Compute the tile shape
  if distributed_group_size == 8:
    tile_shape = (2, 2)
  if distributed_group_size == 16:
    tile_shape = (2, 4)
  if distributed_group_size == 32:
    tile_shape = (4, 4)
  if distributed_group_size == 64:
    tile_shape = (4, 8)
  if tile_shape and input_partition_dims:
    # Cores of a chip are typically used for model parallelism
    tile_shape = tuple(np.array(tile_shape) * np.array((1, 2)))

  return physical_shape, tile_shape


def cross_replica_average(inputs,
                          num_shards,
                          distributed_group_size,
                          physical_shape=None,
                          tile_shape=None,
                          input_partition_dims=None):
  """Calculates the average value of inputs tensor across TPU replicas."""
  if distributed_group_size <= 1:
    return inputs
  group_assignment = None
  if num_shards is not None and distributed_group_size != num_shards:
    if tile_shape is None or physical_shape is None:
      physical_shape, tile_shape = auto_select_shapes(num_shards,
                                                      distributed_group_size,
                                                      input_partition_dims)
    model_cores = 1
    if input_partition_dims:
      model_cores = np.prod(input_partition_dims)
    num_groups = num_shards // distributed_group_size
    if tile_shape and physical_shape:
      if model_cores == 1:
        group_assignment = tile_group_assignment(physical_shape, tile_shape,
                                                 num_groups)
      else:
        group_assignment = sp_tile_group_assignment(physical_shape, tile_shape,
                                                    num_groups)
    else:
      group_size = distributed_group_size
      group_assignment = []
      for g in range(num_shards // group_size):
        replica_ids = [g * group_size + i for i in range(group_size)]
        group_assignment.append(replica_ids)

  outputs = tpu_ops.cross_replica_sum(inputs, group_assignment) / tf.cast(
      distributed_group_size, inputs.dtype)
  return outputs


def distributed_batch_norm(inputs,
                           decay,
                           epsilon,
                           is_training=True,
                           gamma_initializer=None,
                           num_shards=None,
                           distributed_group_size=2,
                           physical_shape=None,
                           tile_shape=None,
                           input_partition_dims=None,
                           scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
    epsilon: Small float added to variance to avoid dividing by zero.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    gamma_initializer:  Initializers for gamma.
    num_shards: Number of shards that participate in the global reduction.
      Default is set to None, that will skip the cross replica sum in and
      normalize across local examples only.
    distributed_group_size: Number of replicas to normalize across in the
      distributed batch normalization.
    physical_shape: shape of the 2D architecture slice, if none it is
      auto selected
    tile_shape: shape of the 2D tile used for the distributed batch
      norm sub-group
    input_partition_dims: shape of the spatial partitoned tile
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.
  """

  with tf.variable_scope(scope, 'batch_normalization', [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined `C` dimension %s.' %
                       (inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta = tf.get_variable(
        'beta',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=True)
    gamma = tf.get_variable(
        'gamma',
        dtype=tf.float32,
        shape=params_shape,
        initializer=gamma_initializer,
        trainable=True)
    # Disable partition setting for moving_mean and moving_variance
    # as assign_moving_average op below doesn't support partitioned variable.
    scope = tf.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf.get_variable(
        'moving_mean',
        shape=params_shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer(),
        trainable=False)
    moving_variance = tf.get_variable(
        'moving_variance',
        shape=params_shape,
        initializer=tf.ones_initializer(),
        trainable=False)
    # Restore scope's partitioner setting.
    scope.set_partitioner(partitioner)

    # Add cross replica sum to do subset mean and variance calculation
    # First compute mean and variance
    if is_training:
      # Execute a distributed batch normalization
      axis = 3
      inputs_dtype = inputs.dtype
      inputs = tf.cast(inputs, tf.float32)
      ndims = len(inputs_shape)
      reduction_axes = [i for i in range(ndims) if i != axis]
      counts, mean_ss, variance_ss, _ = tf.nn.sufficient_statistics(
          inputs, reduction_axes, keep_dims=False)

      model_cores = 1
      if input_partition_dims:
        model_cores = np.prod(input_partition_dims)
      if model_cores > 1:
        mean_variance_ss = tf.concat([mean_ss, variance_ss], 0)
        mean_variance_ss = cross_replica_average(mean_variance_ss, num_shards,
                                                 distributed_group_size,
                                                 physical_shape, tile_shape,
                                                 input_partition_dims)
        num_elements = tf.reduce_prod(mean_ss.get_shape())
        mean_ss = tf.slice(mean_variance_ss, [0], [num_elements])
        variance_ss = tf.slice(mean_variance_ss, [num_elements], [num_elements])
      else:
        mean_ss = cross_replica_average(mean_ss, num_shards,
                                        distributed_group_size, physical_shape,
                                        tile_shape, input_partition_dims)
        variance_ss = cross_replica_average(variance_ss, num_shards,
                                            distributed_group_size,
                                            physical_shape, tile_shape,
                                            input_partition_dims)

      mean, variance = tf.nn.normalize_moments(
          counts, mean_ss, variance_ss, shift=None)
      outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma,
                                          epsilon)
      outputs = tf.cast(outputs, inputs_dtype)
    else:
      mean_ss = cross_replica_average(moving_mean, num_shards, num_shards)
      mean_2ss = cross_replica_average(
          moving_variance + (moving_mean * moving_mean), num_shards, num_shards)
      variance_ss = mean_2ss - (mean_ss * mean_ss)
      outputs, mean, variance = tf.nn.fused_batch_norm(
          inputs,
          gamma,
          beta,
          mean=mean_ss,
          variance=variance_ss,
          epsilon=epsilon,
          is_training=False)

    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean,
          tf.cast(mean, moving_mean.dtype),
          decay,
          zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance,
          tf.cast(variance, moving_variance.dtype),
          decay,
          zero_debias=False)
      tf.add_to_collection('update_ops', update_moving_mean)
      tf.add_to_collection('update_ops', update_moving_variance)

    outputs.set_shape(inputs_shape)
    return outputs
