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
"""Spatial transform functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


def _padding(inputs, paddings, data_format):
  """Pads inputs w.r.t. data format."""
  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs, [[0, 0], [0, 0], paddings, paddings])
  else:
    padded_inputs = tf.pad(
        inputs, [[0, 0], paddings, paddings, [0, 0]])
  return padded_inputs


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
  return _padding(inputs, (pad_beg, pad_end), data_format)


def space_to_depth_fixed_padding(inputs, kernel_size,
                                 data_format='channels_last', block_size=2):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
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
  return _padding(inputs, (pad_beg, pad_end), data_format)


def fused_transpose_and_space_to_depth(
    images, image_size, block_size=2, transpose_input=True):
  """Fuses space-to-depth and transpose.

  Space-to-depth performas the following permutation, which is equivalent to
  tf.nn.space_to_depth.

  images = tf.reshape(images, [batch, h // block_size, block_size,
                               w // block_size, block_size, c])
  images = tf.transpose(images, [0, 1, 3, 2, 4, 5])
  images = tf.reshape(images, [batch, h // block_size, w // block_size,
                               c * (block_size ** 2)])

  Args:
    images: A tensor with a shape of [batch_size, h, w, c] as the images. The
      h and w can be dynamic sizes.
    image_size: A tuple either (short_size, long_size) or (long_size,
      short_size) that represents two shapes of images.
    block_size: A integer for space-to-depth block size.
    transpose_input: A boolean to indicate if the images tensor should be
      transposed.

  Returns:
    A transformed images tensor.

  """
  h, w = image_size
  batch_size, _, _, c = images.get_shape().as_list()
  images = tf.reshape(images,
                      [batch_size, h//block_size, block_size,
                       w//block_size, block_size, c])
  if transpose_input:
    images = tf.transpose(
        images, [1, 3, 0, 2, 4, 5])
    images = tf.reshape(
        images, [h // block_size, w // block_size, batch_size,
                 c * (block_size ** 2)])
    images = tf.reshape(
        images, [-1, batch_size, c * (block_size ** 2)])
  else:
    images = tf.transpose(
        images, [0, 1, 3, 2, 4, 5])
    images = tf.reshape(
        images, [batch_size, h // block_size, w // block_size,
                 c * (block_size ** 2)])
    images = tf.reshape(
        images, [batch_size, -1, c * (block_size ** 2)])
  return images
