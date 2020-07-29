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
"""Mask-RCNN anchor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from REDACTED.tensorflow_models.mlperf.models.rough.mask_rcnn import box_utils
from REDACTED.tensorflow_models.mlperf.models.rough.mask_rcnn import non_max_suppression


def generate_detections_per_image_op(
    cls_outputs, box_outputs, anchor_boxes, image_id, image_info,
    num_detections=100, pre_nms_num_detections=1000, nms_threshold=0.3,
    bbox_reg_weights=(10., 10., 5., 5.)):
  """Generates detections with model outputs and anchors.

  Args:
    cls_outputs: a Tensor with shape [N, num_classes], which stacks class
      logit outputs on all feature levels. The N is the number of total anchors
      on all levels. The num_classes is the number of classes predicted by the
      model. Note that the cls_outputs should be the output of softmax().
    box_outputs: a Tensor with shape [N, num_classes*4], which stacks
      box regression outputs on all feature levels. The N is the number of total
      anchors on all levels.
    anchor_boxes: a Tensor with shape [N, 4], which stacks anchors on all
      feature levels. The N is the number of total anchors on all levels.
    image_id: an integer number to specify the image id.
    image_info: a tensor of shape [5] which encodes the input image's [height,
      width, scale, original_height, original_width]
    num_detections: Number of detections after NMS.
    pre_nms_num_detections: Number of candidates before NMS.
    nms_threshold: a float number to specify the threshold of NMS.
    bbox_reg_weights: a list of 4 float scalars, which are default weights on
      (dx, dy, dw, dh) for normalizing bbox regression targets.
  Returns:
    detections: detection results in a tensor with each row representing
      [image_id, ymin, xmin, ymax, xmax, score, class]
  """
  num_boxes, num_classes = cls_outputs.get_shape().as_list()

  # Removes background class scores.
  cls_outputs = cls_outputs[:, 1:num_classes]
  top_k_scores, top_k_indices_with_classes = tf.nn.top_k(
      tf.reshape(cls_outputs, [-1]),
      k=pre_nms_num_detections,
      sorted=True)
  classes = tf.mod(top_k_indices_with_classes, num_classes - 1)
  top_k_indices = tf.floordiv(top_k_indices_with_classes, num_classes - 1)

  anchor_boxes = tf.gather(anchor_boxes, top_k_indices)
  box_outputs = tf.reshape(
      box_outputs, [num_boxes, num_classes, 4])[:, 1:num_classes, :]
  box_outputs = tf.gather_nd(box_outputs,
                             tf.stack([top_k_indices, classes], axis=1))

  # Applies bounding box regression to anchors.
  boxes = box_utils.batch_decode_box_outputs_op(
      tf.expand_dims(anchor_boxes, axis=0),
      tf.expand_dims(box_outputs, axis=0),
      bbox_reg_weights)[0]
  boxes = box_utils.clip_boxes(
      tf.expand_dims(boxes, axis=0), tf.expand_dims(image_info[:2], axis=0))[0]

  classes = tf.tile(tf.reshape(classes, [1, pre_nms_num_detections]),
                    [num_classes - 1, 1])
  scores = tf.tile(tf.reshape(top_k_scores, [1, pre_nms_num_detections]),
                   [num_classes - 1, 1])
  boxes = tf.tile(tf.reshape(boxes, [1, pre_nms_num_detections, 4]),
                  [num_classes - 1, 1, 1])

  class_bitmask = tf.tile(
      tf.reshape(tf.range(num_classes-1), [num_classes - 1, 1]),
      [1, pre_nms_num_detections])
  scores = tf.where(tf.equal(classes, class_bitmask), scores,
                    tf.zeros_like(scores))
  scores = tf.where(tf.greater(scores, 0.05), scores, tf.zeros_like(scores))
  # Reshape classes to be compartible with the top_k function.
  classes = tf.reshape(classes, [num_classes -1, pre_nms_num_detections, 1])
  scores, sorted_tensors = box_utils.top_k(
      scores, k=pre_nms_num_detections, tensors=[boxes, classes])
  boxes = sorted_tensors[0]
  classes = tf.reshape(sorted_tensors[1],
                       [num_classes - 1, pre_nms_num_detections])

  idx, num_valid = non_max_suppression.non_max_suppression_padded(
      scores, boxes, max_output_size=num_detections,
      iou_threshold=nms_threshold, level=0)

  post_nms_boxes = non_max_suppression.gather_boxes_by_indices(
      boxes, num_detections, idx, num_valid)
  post_nms_scores = non_max_suppression.gather_scores_by_indices(
      scores, num_detections, idx, num_valid)

  # Sorts all results.
  sorted_scores, sorted_indices = tf.nn.top_k(
      tf.to_float(tf.reshape(post_nms_scores, [-1])),
      k=num_detections,
      sorted=True)
  post_nms_boxes = tf.gather(tf.reshape(post_nms_boxes, [-1, 4]),
                             sorted_indices)
  classes = tf.batch_gather(classes, idx)
  post_nms_classes = tf.gather(tf.reshape(classes, [-1]), sorted_indices) + 1

  if isinstance(image_id, int):
    image_id = tf.constant(image_id)
  image_id = tf.reshape(image_id, [])
  detections_result = tf.stack(
      [
          tf.to_float(tf.fill(tf.shape(sorted_scores), image_id)),
          post_nms_boxes[:, 0],
          post_nms_boxes[:, 1],
          post_nms_boxes[:, 2],
          post_nms_boxes[:, 3],
          sorted_scores,
          tf.to_float(post_nms_classes),
      ],
      axis=1)
  return detections_result
