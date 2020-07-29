# Copyright 2018 Google. All Rights Reserved.
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
# Lint as: python3
"""SSD model architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function


from flax import nn
import jax

from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_constants
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax.resnet34 import models


def _class_net(images, level, num_classes, dtype):
  """Class prediction network for SSD.

  Args:
    images: Image tensor.
    level: Integer denoting the ssd level. Level must be in range(3, 8).
    num_classes: Integer denoting the number of object classes including
      background.
    dtype: Floating point type, bfloat16 or float32.

  Returns:
    Tensor as a result of conv layer for the object classification.
  """
  return nn.Conv(
      images,
      num_classes * ssd_constants.NUM_DEFAULTS_BY_LEVEL[level],
      (min(images.shape[1], 3), min(images.shape[2], 3)),
      padding='SAME',
      dtype=dtype,
      kernel_init=jax.nn.initializers.glorot_uniform())


def _box_net(images, level, dtype):
  """Box regression network for SSD.

  Args:
    images: Image tensor.
    level: Integer denoting the ssd level. Level must be in range(3, 8).
    dtype: Floating point type, bfloat16 or float32.

  Returns:
    Tensor as a result of conv layer for the object coordinates.
  """
  return nn.Conv(
      images,
      4 * ssd_constants.NUM_DEFAULTS_BY_LEVEL[level],
      (min(images.shape[1], 3), min(images.shape[2], 3)),
      padding='SAME',
      dtype=dtype,
      kernel_init=jax.nn.initializers.glorot_uniform())


def _rpn_layer(images, filter_sizes, kernel_sizes, strides, paddings, dtype):
  """RPN layers for SSD that applies two conv and two relu ops.

  Args:
    images: Image tensor.
    filter_sizes: A tuple representing filter sizes of 2 consecutive conv.
    kernel_sizes: A tuple representing kernel sizes of 2 consecutive conv.
    strides: A tuple representing strides of 2 consecutive conv.
    paddings: A tuple representing paddings of 2 consecutive conv. A padding
      is either 'SAME' or 'VALID'.
    dtype: Floating point type, bfloat16 or float32.
  Returns:
    Tensor, the result of rpn layer.
  """
  for i in (0, 1):
    tensor = nn.Conv(images,
                     filter_sizes[i],
                     (kernel_sizes[i], kernel_sizes[i]),
                     padding=paddings[i],
                     strides=strides[i],
                     dtype=dtype,
                     kernel_init=jax.nn.initializers.glorot_uniform())
    tensor = nn.relu(tensor)
  return tensor


class SSDModel(nn.Module):
  """SSD Model architecture."""

  def apply(self, x, parameters, axis_name=None, train=True):
    dtype = parameters['dtype']
    min_level = ssd_constants.MIN_LEVEL
    max_level = ssd_constants.MAX_LEVEL
    num_classes = ssd_constants.NUM_CLASSES
    u4 = models.ResNet(
        x,
        num_classes=num_classes,
        parameters=parameters,
        axis_name=axis_name,
        num_layers='SSD-34',
        train=train)
    feats = {}
    feats[3] = u4
    feats[4] = _rpn_layer(feats[3],
                          filter_sizes=(256, 512),
                          kernel_sizes=(1, 3),
                          strides=(None, (2, 2)),
                          paddings=('SAME', 'SAME'),
                          dtype=dtype)

    feats[5] = _rpn_layer(feats[4],
                          filter_sizes=(256, 512),
                          kernel_sizes=(1, 3),
                          strides=(None, (2, 2)),
                          paddings=('SAME', 'SAME'),
                          dtype=dtype)

    feats[6] = _rpn_layer(feats[5],
                          filter_sizes=(128, 256),
                          kernel_sizes=(1, 3),
                          strides=(None, (2, 2)),
                          paddings=('SAME', 'SAME'),
                          dtype=dtype)

    feats[7] = _rpn_layer(feats[6],
                          filter_sizes=(128, 256),
                          kernel_sizes=(1, 3),
                          strides=(None, None),
                          paddings=('SAME', 'VALID'),
                          dtype=dtype)

    feats[8] = _rpn_layer(feats[7],
                          filter_sizes=(128, 256),
                          kernel_sizes=(1, 3),
                          strides=(None, None),
                          paddings=('SAME', 'VALID'),
                          dtype=dtype)

    class_outputs = {}
    box_outputs = {}
    for level in range(min_level, max_level + 1):
      class_outputs[level] = _class_net(feats[level],
                                        level,
                                        num_classes,
                                        dtype)
      box_outputs[level] = _box_net(feats[level],
                                    level,
                                    dtype)
    return class_outputs, box_outputs
