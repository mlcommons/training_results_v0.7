# Lint as: python3
"""Flax implementation of ResNet V1.
"""


from flax import nn
import jax
import jax.numpy as jnp
from jax.util import partial

import numpy as np
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_constants


def _find_replica_groups(global_x, inner_x, inner_y, outer_group_size):
  """Find replica groups for SPMD."""
  groups_x = global_x // inner_x
  inner_replica_groups = []
  for group_id in range(outer_group_size):
    sub_group_ids = []
    row_group_id = group_id // groups_x
    col_group_id = group_id % groups_x
    starting_id = row_group_id * inner_y * global_x + col_group_id * inner_x
    for y in range(inner_y):
      inner_group_ids = list(range(starting_id, starting_id + inner_x))
      if y % 2 == 1:
        inner_group_ids.reverse()
      sub_group_ids.extend(inner_group_ids)
      starting_id += global_x
    inner_replica_groups.append(sub_group_ids)
  return inner_replica_groups


def _make_replica_groups(parameters):
  """Construct local nearest-neighbor rings given the JAX device assignment."""
  if 'bn_group_size' not in parameters or parameters['bn_group_size'] <= 1:
    return None

  inner_group_size = parameters['bn_group_size']
  world_size = parameters['num_replicas']
  if parameters['num_partitions'] > 1 and not parameters['enable_wus']:
    # TODO(b/158151451): find better inner_replicia_groups.
    group_perm = [i * 2 for i in range(inner_group_size // 2)] + [
        i * 2 + 1 for i in range(inner_group_size // 2 - 1, -1, -1)
    ]
    inner_replica_groups = []
    for g in range(world_size // inner_group_size):
      replica_ids = [g * inner_group_size + i for i in group_perm]
      inner_replica_groups.append(replica_ids)
    return inner_replica_groups

  outer_group_size, ragged = divmod(world_size, inner_group_size)
  assert not ragged, 'inner group size must evenly divide global device count'
  # the last device should have maximal x and y coordinate
  def bounds_from_last_device(device):
    x, y, z = device.coords
    return (x + 1) * (device.core_on_chip + 1), (y + 1) * (z + 1)

  global_x, _ = bounds_from_last_device(jax.devices()[-1])
  per_host_x, per_host_y = bounds_from_last_device(jax.local_devices(0)[-1])

  if parameters['num_partitions'] > 1:
    global_x = global_x // (parameters['num_partitions'] // 2)

  # host_x, hosts_y = global_x // per_host_x, global_y // per_host_y
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

    if parameters['num_partitions'] > 1:
      return _find_replica_groups(global_x, inner_x, inner_y, outer_group_size)

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


class SpaceToDepthConv(nn.base.Module):
  """Convolution Module wrapping lax.conv_general_dilated."""

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            input_dilation=None,
            kernel_dilation=None,
            feature_group_count=1,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=nn.linear.default_kernel_init,
            bias_init=nn.initializers.zeros):
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      input_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`.
        Convolution with input dilation `d` is equivalent to transposed
        convolution with stride `d`.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, dtype)

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    # Create the conv0 kernel w.r.t. the original image size.
    # (no space-to-depth).
    filters = features
    original_kernel_size = (kernel_size, kernel_size)
    space_to_depth_block_size = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    batch_size, h, w, channel = inputs.shape

    original_input_shape = (
        batch_size, h * space_to_depth_block_size,
        w * space_to_depth_block_size, 3)
    in_features = original_input_shape[-1]
    assert in_features % feature_group_count == 0
    kernel_shape = original_kernel_size + (in_features // feature_group_count,
                                           filters)
    kernel = self.param('kernel', kernel_shape, kernel_init)
    kernel = jnp.asarray(kernel, dtype)
    # [7, 7, 3, 64] --> [8, 8, 3, 64]
    kernel = jnp.pad(
        kernel,
        pad_width=[[1, 0], [1, 0], [0, 0], [0, 0]],
        mode='constant',
        constant_values=0.)
    # Transform kernel follows the space-to-depth logic:
    # http://shortn/_9YvHW96xPJ
    kernel = jnp.reshape(
        kernel,
        [4, space_to_depth_block_size,
         4, space_to_depth_block_size, 3, filters])
    kernel = jnp.transpose(kernel, [0, 2, 1, 3, 4, 5])
    kernel = jnp.reshape(kernel, [4, 4, int(channel), filters])
    kernel = kernel.astype(inputs.dtype)

    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint: disable=protected-access

    y = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=kernel,
        window_strides=(1, 1),
        padding='VALID',
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=None)
    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias
    return y


def fixed_padding(inputs, kernel_size, data_format='channels_last'):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg
  if data_format == 'channels_first':
    padded_inputs = jnp.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = jnp.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def space_to_depth_fixed_padding(inputs,
                                 kernel_size,
                                 data_format='channels_last',
                                 block_size=2):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or `[batch,
      height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
      operations. Should be a positive integer.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    block_size: `int` block size for space-to-depth convolution.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  pad_total = kernel_size - 1
  pad_beg = (pad_total // 2 + 1) // block_size
  pad_end = (pad_total // 2) // block_size
  if data_format == 'channels_first':
    padded_inputs = jnp.pad(
        inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = jnp.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

  return padded_inputs


def conv2d_fixed_padding(inputs,
                         filters,
                         kernel_size,
                         strides,
                         data_format='channels_last',
                         dtype=jnp.float32,
                         name=''):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
    dtype: Data type for the convolution. e.g.,jnp.float32 or jnp.bfloat16
    name: `String` name for the convolution.
  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)

  return nn.Conv(
      inputs,
      filters,
      (kernel_size, kernel_size),
      strides=(strides, strides),
      padding=('SAME' if strides == 1 else 'VALID'),
      bias=False,
      dtype=dtype,
      name=name)


class ResidualBlock(nn.Module):
  """Residual ResNet block."""

  def apply(self, x, filters, parameters, strides=1, train=True, axis_name=None,
            use_projection=False, data_format='channels_last'):
    dtype = parameters['dtype']
    replica_groups = _make_replica_groups(parameters)
    batch_norm = nn.BatchNorm.partial(
        use_running_average=not train, momentum=0.9, epsilon=1e-5,
        axis_name=axis_name, dtype=dtype, axis_index_groups=replica_groups)

    conv = partial(conv2d_fixed_padding, dtype=dtype, data_format=data_format)
    residual = x
    if use_projection:
      residual = conv(inputs=residual, filters=filters, kernel_size=1,
                      strides=strides, name='proj_conv')
      residual = batch_norm(residual, name='proj_bn')

    y = conv(inputs=x, filters=filters, kernel_size=3, strides=strides,
             name='conv1')
    y = batch_norm(y, name='bn2')
    y = nn.relu(y)
    y = conv(inputs=y, filters=filters, kernel_size=3, strides=1, name='conv2')

    y = batch_norm(y, name='bn3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)
    return y


def func_conv0_space_to_depth(inputs, data_format='channels_last',
                              dtype=jnp.float32):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, height_in, width_in, channels]`.
    data_format: `str` either "channels_first" for `[batch, channels, height,
      width]` or "channels_last for `[batch, height, width, channels]`.
    dtype: Data type jnp.float32 or jnp.bfloat16.
  Returns:
    A `Tensor` with the same type as `inputs`.
  """

  # Create the conv0 kernel w.r.t. the original image size. (no space-to-depth).
  filters = 64
  kernel_size = 7
  space_to_depth_block_size = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE

  inputs = space_to_depth_fixed_padding(inputs, kernel_size, data_format,
                                        space_to_depth_block_size)
  return SpaceToDepthConv(
      inputs,
      filters,
      kernel_size=kernel_size,
      strides=(1, 1),
      padding='VALID',
      bias=False,
      dtype=dtype,
      name='init_conv')


class ResNet(nn.Module):
  """ResNetV1 modified for SSD excluding dense layer and softmax."""

  def apply(self, x, num_classes, parameters, num_filters=64,
            train=True, axis_name=None, num_layers='34'):
    block_sizes = [3, 4, 6]
    data_format = 'channels_last'
    if ('conv0_space_to_depth' in parameters and
        parameters['conv0_space_to_depth']):
      # conv0 uses space-to-depth transform for TPU performance.
      x = func_conv0_space_to_depth(inputs=x, data_format=data_format,
                                    dtype=parameters['dtype'])
    else:
      x = conv2d_fixed_padding(
          inputs=x,
          filters=num_filters,
          kernel_size=7,
          strides=2,
          data_format=data_format,
          name='init_conv')

    replica_groups = _make_replica_groups(parameters)
    x = nn.BatchNorm(
        x,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        name='init_bn',
        axis_name=axis_name,
        dtype=parameters['dtype'],
        axis_index_groups=replica_groups)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = 2 if i == 1 and j == 0 else 1
        use_projection = False if i == 0 or j > 0 else True
        x = ResidualBlock(
            x,
            num_filters * 2**i,
            parameters,
            strides=strides,
            train=train,
            axis_name=axis_name,
            use_projection=use_projection,
            data_format=data_format)
    if num_layers == '34':
      x = jnp.mean(x, axis=(1, 2))
      x = nn.Dense(x, num_classes, kernel_init=nn.initializers.normal(),
                   dtype=jnp.float32)  # TODO(deveci): dtype=dtype
      x = nn.log_softmax(x)
    return x


# a dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    '34': [3, 4, 6, 3],  # Option to use to train resnet34 from scratch to
                         # generate checkpoints.
    'SSD-34': [3, 4, 6],  # Option to use to when calling from ssd.
}


class FakeResNet(nn.Module):
  """Fake model for testing."""

  def apply(self,
            x,
            num_classes,
            train=True,
            batch_stats=None,
            axis_name=None,
            dtype=jnp.float32):
    x = nn.BatchNorm(x,
                     batch_stats=batch_stats,
                     use_running_average=not train,
                     momentum=0.9, epsilon=1e-5,
                     name='init_bn', axis_name=axis_name, dtype=dtype)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes, kernel_init=nn.initializers.normal(),
                 dtype=dtype)
    x = nn.log_softmax(x)
    return x
