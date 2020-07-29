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
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from REDACTED.tensorflow_models.mlperf.models.rough.util import image_util


def conv2d_space_to_depth(inputs, filters, kernel_size, strides, block_size=2):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, height_in, width_in, channels]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    block_size: space-to-depth padding block-size

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """

  def _round_up(kernel_size, block_size):
    remainder = kernel_size % block_size
    if remainder == 0:
      return block_size
    else:
      return kernel_size + block_size - remainder

  padded_k = _round_up(kernel_size, block_size)
  _, _, _, in_f = inputs.get_shape().as_list()

  # Create the padded kernel
  kernel = tf.get_variable(
      'kernel_conv2d_opt',
      shape=[padded_k, padded_k, in_f // 4, filters],
      initializer=tf.variance_scaling_initializer(),
      trainable=True,
      dtype=tf.float32)

  # Zero padded region
  mx = tf.constant([1, 1, 1, 1, 1, 1, 1, 0], dtype=tf.float32)
  kernel = kernel * tf.reshape(mx, [1, padded_k, 1, 1]) * tf.reshape(
      mx, [padded_k, 1, 1, 1])

  # Transpose to enable space-to-depth optimization
  kernel = tf.reshape(kernel, [
      padded_k // 2, block_size, padded_k // 2, block_size, in_f // 4, filters
  ])
  kernel = tf.transpose(kernel, [0, 2, 1, 3, 4, 5])
  kernel = tf.reshape(kernel, [padded_k // 2, padded_k // 2, in_f, filters])
  kernel = tf.cast(kernel, inputs.dtype)

  # Add a convolution operation using this kernel
  s = strides // 2
  return tf.nn.conv2d(
      input=inputs,
      filter=kernel,
      strides=[s, s, s, s],
      padding='SAME',
      data_format='NHWC')


def conv2d_norm_relu(inputs,
                     filters,
                     kernel_size,
                     strides,
                     is_training,
                     relu=True,
                     init_zero=False,
                     space_to_depth=False,
                     num_replicas=1,
                     distributed_group_size=1,
                     batch_norm_decay=0.9,
                     batch_norm_epsilon=1e-5):
  """conv2d followed by batchnorm and relu.

  Args:
    inputs: `Tensor` of size `[batch, height_in, width_in, channels]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
      normalization with 0 instead of 1 (default).
    space_to_depth: If True, we use the space-to-depth transformation.
    num_replicas: Number of replicas that participate in the global reduction.
      Default is set to None, that will skip the cross replica sum in and
      normalize across local examples only.
    distributed_group_size: Number of replicas to normalize across in the
      distributed batch normalization.
    batch_norm_decay: 'float' the decay rate of the BN EMA.
    batch_norm_epsilon: 'float' the epsilon used in the BN denominator.

  Returns:
    A `Tensor` of shape `[batch, height_out, width_out, filters]`.
  """
  if space_to_depth:
    inputs = conv2d_space_to_depth(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides)
    inputs = tf.identity(inputs, 'initial_conv')
  else:
    if strides > 1:
      pad_total = kernel_size - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      inputs = tf.pad(inputs,
                      [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer())

  initializer = tf.zeros_initializer() if init_zero else tf.ones_initializer()

  # TODO(b/147558625): remove the branch to always go to True path
  if distributed_group_size > 0:
    inputs = image_util.distributed_batch_norm(
        inputs=inputs,
        decay=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        is_training=is_training,
        gamma_initializer=initializer,
        num_shards=num_replicas,
        distributed_group_size=distributed_group_size)
  else:
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=3,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        center=True,
        scale=True,
        training=is_training,
        fused=True,
        gamma_initializer=initializer)

  if relu:
    inputs = tf.nn.relu(inputs)

  return inputs


def resnet_v1(
    resnet_depth,
    num_classes,
    use_space_to_depth=False,
    num_replicas=1,
    distributed_group_size=1,
    batch_norm_decay=0.9,
    batch_norm_epsilon=1e-5):
  """Returns the ResNet model for a given size and number of output classes."""

  def model(inputs, is_training):
    """Creation of the model graph."""
    model_params = {
        18: [False, [2, 2, 2, 2]],
        34: [False, [3, 4, 6, 3]],
        50: [True, [3, 4, 6, 3]],
        101: [True, [3, 4, 23, 3]],
        152: [True, [3, 8, 36, 3]],
        200: [True, [3, 24, 36, 3]]
    }
    use_bottleneck, layers = model_params[resnet_depth]
    inputs = conv2d_norm_relu(
        inputs, 64, 7, 2, is_training, space_to_depth=use_space_to_depth,
        num_replicas=num_replicas,
        distributed_group_size=distributed_group_size,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    inputs = tf.layers.max_pooling2d(
        inputs=inputs, pool_size=3, strides=2, padding='SAME')

    for i in range(4):
      filters = 64 * pow(2, i)
      filters_out = 4 * filters if use_bottleneck else filters
      for layer in range(layers[i]):
        strides = 1 if (i == 0 or layer > 0) else 2
        shortcut = inputs
        if layer == 0:
          shortcut = conv2d_norm_relu(
              inputs, filters_out, 1, strides, is_training, False,
              num_replicas=num_replicas,
              distributed_group_size=distributed_group_size,
              batch_norm_decay=batch_norm_decay,
              batch_norm_epsilon=batch_norm_epsilon)

        if use_bottleneck:
          inputs = conv2d_norm_relu(
              inputs, filters, 1, 1, is_training,
              num_replicas=num_replicas,
              distributed_group_size=distributed_group_size,
              batch_norm_decay=batch_norm_decay,
              batch_norm_epsilon=batch_norm_epsilon)
          inputs = conv2d_norm_relu(
              inputs, filters, 3, strides, is_training,
              num_replicas=num_replicas,
              distributed_group_size=distributed_group_size,
              batch_norm_decay=batch_norm_decay,
              batch_norm_epsilon=batch_norm_epsilon)
          inputs = conv2d_norm_relu(
              inputs, filters_out, 1, 1, is_training,
              False, True,
              num_replicas=num_replicas,
              distributed_group_size=distributed_group_size,
              batch_norm_decay=batch_norm_decay,
              batch_norm_epsilon=batch_norm_epsilon)
        else:
          inputs = conv2d_norm_relu(
              inputs, filters, 3, strides, is_training,
              num_replicas=num_replicas,
              distributed_group_size=distributed_group_size,
              batch_norm_decay=batch_norm_decay,
              batch_norm_epsilon=batch_norm_epsilon)
          inputs = conv2d_norm_relu(
              inputs, filters_out, 3, 1, is_training,
              False, True,
              num_replicas=num_replicas,
              distributed_group_size=distributed_group_size,
              batch_norm_decay=batch_norm_decay,
              batch_norm_epsilon=batch_norm_epsilon)
        inputs = tf.nn.relu(inputs + shortcut)

    inputs = tf.reshape(inputs, [inputs.shape[0], -1, inputs.shape[-1]])
    inputs = tf.reduce_mean(inputs, 1)
    inputs = tf.layers.dense(
        inputs=inputs,
        units=num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    return inputs

  return model
