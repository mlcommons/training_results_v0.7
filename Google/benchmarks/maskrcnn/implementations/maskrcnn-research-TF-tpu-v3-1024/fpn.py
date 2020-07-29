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
"""Mask-RCNN (via ResNet) model definition.

Uses the ResNet model as a basis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from REDACTED.mask_rcnn import resnet


_RESNET_MAX_LEVEL = 5


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A tensor with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    A tensor with a shape of [batch, height_in*scale, width_in*scale, channels].
    Same dtype as input data.
  """
  with tf.name_scope('nearest_upsampling'):
    bs, h, w, c = data.get_shape().as_list()
    bs = -1 if bs is None else bs
    # Use reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def resnet_fpn(features,
               min_level=3,
               max_level=7,  # pylint: disable=unused-argument
               resnet_depth=50,
               conv0_kernel_size=7,
               conv0_space_to_depth_block_size=0,
               is_training_bn=False):
  """ResNet feature pyramid networks."""
  # upward layers
  with tf.variable_scope('resnet%s' % resnet_depth):
    resnet_fn = resnet.resnet_v1(resnet_depth, conv0_kernel_size,
                                 conv0_space_to_depth_block_size)
    u2, u3, u4, u5 = resnet_fn(features, is_training_bn)

  feats_bottom_up = {
      2: u2,
      3: u3,
      4: u4,
      5: u5,
  }

  with tf.variable_scope('resnet_fpn'):
    # lateral connections
    feats_lateral = {}
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats_lateral[level] = tf.layers.conv2d(
          feats_bottom_up[level],
          filters=256,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)

    # add top-down path
    feats = {_RESNET_MAX_LEVEL: feats_lateral[_RESNET_MAX_LEVEL]}
    for level in range(_RESNET_MAX_LEVEL - 1, min_level - 1, -1):
      feats[level] = nearest_upsampling(
          feats[level + 1], 2) + feats_lateral[level]

    # add post-hoc 3x3 convolution kernel
    for level in range(min_level, _RESNET_MAX_LEVEL + 1):
      feats[level] = tf.layers.conv2d(
          feats[level],
          filters=256,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)

    # Use original FPN P6 level implementation from CVPR'17 FPN paper instead of
    # coarse FPN levels introduced for RetinaNet.
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/FPN.py  # pylint: disable=line-too-long
    feats[6] = tf.layers.max_pooling2d(
        inputs=feats[5],
        pool_size=1,
        strides=2,
        padding='valid',
        name='p6')

  return feats
