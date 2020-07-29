# Lint as: python3
"""Flax implementation of ResNet V1.
"""


from flax import nn

from jax import lax
import jax.numpy as jnp


class SpaceToDepthConv(nn.base.Module):
  """Convolution on space-to-depth transformed input images."""

  def apply(self,
            inputs,
            filters,
            kernel_size,
            block_size,
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
      filters: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      block_size: shape of space-to-depth blocks.
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
      strides = block_size
    assert strides[0] % block_size[0] == 0
    assert strides[1] % block_size[1] == 0
    strides = tuple(s // b for s, b in zip(strides, block_size))

    # create kernel as if there were no space to depth
    batch_size, h, w, features = inputs.shape
    original_input_shape = (
        batch_size, h * block_size[0],
        w * block_size[1], features // block_size[0] // block_size[1])
    in_features = original_input_shape[-1]
    assert in_features % feature_group_count == 0
    kernel_shape = kernel_size + (in_features // feature_group_count, filters)
    kernel = self.param('kernel', kernel_shape, kernel_init)
    kernel = jnp.asarray(kernel, dtype)

    # zero-pad kernel to multiple of block size (e.g. 7x7 --> 8x8)
    h_blocks, h_ragged = divmod(kernel_size[0], block_size[0])
    h_blocks = h_blocks + 1
    if h_ragged != 0:
      kernel = jnp.pad(
          kernel,
          pad_width=[[block_size[0] - h_ragged, 0], [0, 0], [0, 0], [0, 0]],
          mode='constant',
          constant_values=0.)
    w_blocks, w_ragged = divmod(kernel_size[1], block_size[1])
    w_blocks = w_blocks + 1
    if w_ragged != 0:
      kernel = jnp.pad(
          kernel,
          pad_width=[[0, 0], [block_size[1] - w_ragged, 0], [0, 0], [0, 0]],
          mode='constant',
          constant_values=0.)

    # transform kernel following space-to-depth logic: http://shortn/_9YvHW96xPJ
    kernel = jnp.reshape(
        kernel,
        [h_blocks, block_size[0],
         w_blocks, block_size[1], in_features // feature_group_count, filters])
    kernel = jnp.transpose(kernel, [0, 2, 1, 3, 4, 5])
    kernel = jnp.reshape(kernel, [h_blocks, w_blocks, features, filters])
    kernel = kernel.astype(inputs.dtype)

    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint: disable=protected-access

    y = lax.conv_general_dilated(
        lhs=inputs,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision)
    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias
    return y


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, filters, strides=(1, 1), train=True, axis_name=None,
            axis_index_groups=None, dtype=jnp.float32):
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = nn.BatchNorm.partial(
        use_running_average=not train, momentum=0.9, epsilon=1e-5,
        axis_name=axis_name, axis_index_groups=axis_index_groups, dtype=dtype)
    conv = nn.Conv.partial(bias=False, dtype=dtype)

    residual = x
    if needs_projection:
      residual = conv(residual, filters * 4, (1, 1), strides, name='proj_conv')
      residual = batch_norm(residual, name='proj_bn')

    y = conv(x, filters, (1, 1), name='conv1')
    y = batch_norm(y, name='bn1')
    y = nn.relu(y)
    y = conv(y, filters, (3, 3), strides, name='conv2')
    y = batch_norm(y, name='bn2')
    y = nn.relu(y)
    y = conv(y, filters * 4, (1, 1), name='conv3')

    y = batch_norm(y, name='bn3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""

  def apply(self, x, num_classes, num_filters=64, num_layers=50,
            train=True, axis_name=None, axis_index_groups=None,
            dtype=jnp.float32, conv0_space_to_depth=False):
    if num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[num_layers]
    if conv0_space_to_depth:
      conv = SpaceToDepthConv.partial(block_size=(2, 2),
                                      padding=[(2, 1), (2, 1)])
    else:
      conv = nn.Conv.partial(padding=[(3, 3), (3, 3)])
    x = conv(x, num_filters, kernel_size=(7, 7), strides=(2, 2), bias=False,
             dtype=dtype, name='conv0')
    x = nn.BatchNorm(x,
                     use_running_average=not train,
                     momentum=0.9, epsilon=1e-5,
                     name='init_bn', axis_name=axis_name,
                     axis_index_groups=axis_index_groups, dtype=dtype)
    x = nn.relu(x)  # MLPerf-required
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(x, num_filters * 2 ** i,
                          strides=strides,
                          train=train, axis_name=axis_name,
                          axis_index_groups=axis_index_groups, dtype=dtype)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes, kernel_init=nn.initializers.normal(),
                 dtype=dtype)
    x = nn.log_softmax(x)
    return x


# a dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


class FakeResNet(nn.Module):
  """Minimal NN (for debugging) with the same signature as a ResNet."""

  def apply(self, x, num_classes, train=True, batch_stats=None, axis_name=None,
            axis_index_groups=None, dtype=jnp.float32):
    x = nn.BatchNorm(x,
                     batch_stats=batch_stats,
                     use_running_average=not train,
                     momentum=0.9, epsilon=1e-5,
                     name='init_bn', axis_name=axis_name,
                     axis_index_groups=axis_index_groups, dtype=dtype)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes, kernel_init=nn.initializers.normal(),
                 dtype=dtype)
    x = nn.log_softmax(x)
    return x
