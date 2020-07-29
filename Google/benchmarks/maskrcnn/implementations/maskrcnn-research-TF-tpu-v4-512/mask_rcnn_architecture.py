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

from REDACTED.tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from REDACTED.mask_rcnn import box_utils
from REDACTED.mask_rcnn import core_assignment_utils
from REDACTED.mask_rcnn import non_max_suppression
from REDACTED.mask_rcnn.object_detection import balanced_positive_negative_sampler


_EPSILON = 1e-8


def _add_class_assignments(iou, scaled_gt_boxes, gt_labels):
  """Computes object category assignment for each box.

  Args:
    iou: a tensor for the iou matrix with a shape of
      [batch_size, K, MAX_NUM_INSTANCES]. K is the number of post-nms RoIs
      (i.e., rpn_post_nms_topn).
    scaled_gt_boxes: a tensor with a shape of
      [batch_size, MAX_NUM_INSTANCES, 4]. This tensor might have paddings with
      negative values. The coordinates of gt_boxes are in the pixel coordinates
      of the scaled image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
  Returns:
    max_boxes: a tensor with a shape of [batch_size, K, 4], representing
      the ground truth coordinates of each roi.
    max_classes: a int32 tensor with a shape of [batch_size, K], representing
      the ground truth class of each roi.
    max_overlap: a tensor with a shape of [batch_size, K], representing
      the maximum overlap of each roi.
    argmax_iou: a tensor with a shape of [batch_size, K], representing the iou
      argmax.
  """
  with tf.name_scope('add_class_assignments'):
    batch_size, _, _ = iou.get_shape().as_list()
    argmax_iou = tf.argmax(iou, axis=2, output_type=tf.int32)
    indices = tf.reshape(
        argmax_iou + tf.expand_dims(
            tf.range(batch_size) * tf.shape(gt_labels)[1], 1), [-1])
    max_classes = tf.reshape(
        tf.gather(tf.reshape(gt_labels, [-1, 1]), indices), [batch_size, -1])
    max_overlap = tf.reduce_max(iou, axis=2)
    bg_mask = tf.equal(max_overlap, tf.zeros_like(max_overlap))
    max_classes = tf.where(bg_mask, tf.zeros_like(max_classes), max_classes)

    max_boxes = tf.reshape(
        tf.gather(tf.reshape(scaled_gt_boxes, [-1, 4]), indices),
        [batch_size, -1, 4])
    max_boxes = tf.where(
        tf.tile(tf.expand_dims(bg_mask, axis=2), [1, 1, 4]),
        tf.zeros_like(max_boxes), max_boxes)
  return max_boxes, max_classes, max_overlap, argmax_iou


def encode_box_targets(boxes, gt_boxes, gt_labels, bbox_reg_weights):
  """Encodes predicted boxes with respect to ground truth boxes."""
  with tf.name_scope('encode_box_targets'):
    box_targets = box_utils.batch_encode_box_targets_op(
        boxes, gt_boxes, bbox_reg_weights)
    # If a target is background, the encoded box target should be zeros.
    mask = tf.tile(
        tf.expand_dims(tf.equal(gt_labels, tf.zeros_like(gt_labels)), axis=2),
        [1, 1, 4])
    box_targets = tf.where(mask, tf.zeros_like(box_targets), box_targets)
  return box_targets


def proposal_label_op(boxes, gt_boxes, gt_labels, image_info,
                      batch_size_per_im=512, fg_fraction=0.25, fg_thresh=0.5,
                      bg_thresh_hi=0.5, bg_thresh_lo=0.):
  """Assigns the proposals with ground truth labels and performs subsmpling.

  Given proposal `boxes`, `gt_boxes`, and `gt_labels`, the function uses the
  following algorithm to generate the final `batch_size_per_im` RoIs.
  1. Calculates the IoU between each proposal box and each gt_boxes.
  2. Assigns each proposal box with a ground truth class and box label by
     choosing the largest overlap.
  3. Samples `batch_size_per_im` boxes from all proposal boxes, and returns
     box_targets, class_targets, and RoIs.
  The reference implementations of #1 and #2 are here: https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py  # pylint: disable=line-too-long
  The reference implementation of #3 is here: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py.  # pylint: disable=line-too-long

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates of scaled images in
      [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a value of -1. The coordinates of gt_boxes
      are in the pixel coordinates of the original image scale.
    gt_labels: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES]. This
      tensor might have paddings with a value of -1.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width.
    batch_size_per_im: a integer represents RoI minibatch size per image.
    fg_fraction: a float represents the target fraction of RoI minibatch that
      is labeled foreground (i.e., class > 0).
    fg_thresh: a float represents the overlap threshold for an RoI to be
      considered foreground (if >= fg_thresh).
    bg_thresh_hi: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
    bg_thresh_lo: a float represents the overlap threshold for an RoI to be
      considered background (class = 0 if overlap in [LO, HI)).
  Returns:
    box_targets: a tensor with a shape of [batch_size, K, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi. K is the number of sample RoIs (e.g., batch_size_per_im).
    class_targets: a integer tensor with a shape of [batch_size, K]. The tensor
      contains the ground truth class for each roi.
    rois: a tensor with a shape of [batch_size, K, 4], representing the
      coordinates of the selected RoI.
    proposal_to_label_map: a tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels. proposal_to_label_map[i]
      means the index of the ground truth instance for the i-th proposal.
  """
  with tf.name_scope('proposal_label'):
    batch_size = boxes.shape[0]
    # Scales ground truth boxes to the scaled image coordinates.
    image_scale = 1 / image_info[:, 2]
    scaled_gt_boxes = gt_boxes * tf.reshape(image_scale, [batch_size, 1, 1])

    # The reference implementation intentionally includes ground truth boxes in
    # the proposals. see https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/json_dataset.py#L359.  # pylint: disable=line-too-long
    boxes = tf.concat([boxes, scaled_gt_boxes], axis=1)
    iou = box_utils.bbox_overlap(boxes, scaled_gt_boxes)

    (pre_sample_box_targets, pre_sample_class_targets, max_overlap,
     proposal_to_label_map) = _add_class_assignments(
         iou, scaled_gt_boxes, gt_labels)

    # Generates a random sample of RoIs comprising foreground and background
    # examples. reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/fast_rcnn.py#L132  # pylint: disable=line-too-long
    positives = tf.greater(max_overlap,
                           fg_thresh * tf.ones_like(max_overlap))
    negatives = tf.logical_and(
        tf.greater_equal(max_overlap,
                         bg_thresh_lo * tf.ones_like(max_overlap)),
        tf.less(max_overlap,
                bg_thresh_hi * tf.ones_like(max_overlap)))
    pre_sample_class_targets = tf.where(
        negatives, tf.zeros_like(pre_sample_class_targets),
        pre_sample_class_targets)
    proposal_to_label_map = tf.where(
        negatives, tf.zeros_like(proposal_to_label_map),
        proposal_to_label_map)

    # Handles ground truth paddings.
    ignore_mask = tf.less(
        tf.reduce_min(iou, axis=2), tf.zeros_like(max_overlap))
    # indicator includes both positive and negative labels.
    # labels includes only positives labels.
    # positives = indicator & labels.
    # negatives = indicator & !labels.
    # ignore = !indicator.
    labels = positives
    pos_or_neg = tf.logical_or(positives, negatives)
    indicator = tf.logical_and(pos_or_neg, tf.logical_not(ignore_mask))

    all_samples = []
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            positive_fraction=fg_fraction, is_static=True))
    # Batch-unroll the sub-sampling process.
    for i in range(batch_size):
      samples = sampler.subsample(
          indicator[i], batch_size_per_im, labels[i])
      all_samples.append(samples)
    all_samples = tf.stack([all_samples], axis=0)[0]
    # A workaround to get the indices from the boolean tensors.
    _, samples_indices = tf.nn.top_k(tf.to_int32(all_samples),
                                     k=batch_size_per_im, sorted=True)
    # Contructs indices for gather.
    samples_indices = tf.reshape(
        samples_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(boxes)[1], 1), [-1])
    rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), samples_indices),
        [batch_size, -1, 4])
    class_targets = tf.reshape(
        tf.gather(
            tf.reshape(pre_sample_class_targets, [-1, 1]), samples_indices),
        [batch_size, -1])
    sample_box_targets = tf.reshape(
        tf.gather(tf.reshape(pre_sample_box_targets, [-1, 4]), samples_indices),
        [batch_size, -1, 4])
    sample_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), samples_indices),
        [batch_size, -1])
  return sample_box_targets, class_targets, rois, sample_proposal_to_label_map


def _proposal_op_per_level(scores, boxes, anchor_boxes, image_info,
                           rpn_pre_nms_topn, rpn_post_nms_topn,
                           rpn_nms_threshold, rpn_min_size, level):
  """Proposes RoIs for the second stage nets.

  This proposal op performs the following operations.
    1. for each location i in a (H, W) grid:
         generate A anchor boxes centered on cell i
         apply predicted bbox deltas to each of the A anchors at cell i
    2. clip predicted boxes to image
    3. remove predicted boxes with either height or width < threshold
    4. sort all (proposal, score) pairs by score from highest to lowest
    5. take the top rpn_pre_nms_topn proposals before NMS
    6. apply NMS with a loose threshold (0.7) to the remaining proposals
    7. take after_nms_topN proposals after NMS
    8. return the top proposals
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/generate_proposals.py  # pylint: disable=line-too-long

  Args:
    scores: a tensor with a shape of
      [batch_size, height, width, num_anchors].
    boxes: a tensor with a shape of
      [batch_size, height, width, num_anchors * 4], in the encoded form.
    anchor_boxes: an Anchors object that contains the anchors with a shape of
      [batch_size, height, width, num_anchors * 4].
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
    level: a integer number for the level that the function operates on.
  Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals. It has same dtype as input
      scores.
    boxes: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      represneting the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax]. It has same dtype as
      input boxes.

  """
  with tf.name_scope('proposal-l%d' % level):
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take the top rpn_pre_nms_topn proposals before NMS
    batch_size, h, w, num_anchors = scores.get_shape().as_list()
    scores = tf.reshape(scores, [batch_size, -1])
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    # Map scores to [0, 1] for convenince of setting min score.
    scores = tf.sigmoid(scores)

    topk_limit = (h * w * num_anchors if h * w * num_anchors < rpn_pre_nms_topn
                  else rpn_pre_nms_topn)
    anchor_boxes = tf.reshape(anchor_boxes, [batch_size, -1, 4])
    scores, boxes_list = box_utils.top_k(scores, k=topk_limit,
                                         tensors=[boxes, anchor_boxes])
    boxes = boxes_list[0]
    anchor_boxes = boxes_list[1]

    # Transforms anchors into proposals via bbox transformations.
    boxes = box_utils.batch_decode_box_outputs_op(anchor_boxes, boxes)

    # 2. clip proposals to image (may result in proposals with zero area
    # that will be removed in the next step)
    boxes = box_utils.clip_boxes(boxes, image_info[:, :2])

    # 3. remove predicted boxes with either height or width < min_size
    scores, boxes = box_utils.filter_boxes(scores, boxes, rpn_min_size,
                                           image_info)

    # 6. apply loose nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    if topk_limit < rpn_post_nms_topn:
      post_nms_topk_limit = topk_limit
    else:
      post_nms_topk_limit = rpn_post_nms_topn
    if rpn_nms_threshold > 0:
      idx, num_valid = non_max_suppression.non_max_suppression_padded(
          scores, boxes, max_output_size=post_nms_topk_limit,
          iou_threshold=rpn_nms_threshold, level=level)

      scores = non_max_suppression.gather_scores_by_indices(
          scores, post_nms_topk_limit, idx, num_valid)
      boxes = non_max_suppression.gather_boxes_by_indices(
          boxes, post_nms_topk_limit, idx, num_valid)

    scores, boxes = box_utils.top_k(
        scores, k=post_nms_topk_limit, tensors=[boxes])
    boxes = boxes[0]

    return scores, boxes


def proposal_op(scores_outputs, box_outputs, all_anchors, image_info,
                rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
                rpn_min_size):
  """Proposes RoIs for the second stage nets.

  This proposal op performs the following operations.
    1. propose rois at each level.
    2. collect all proposals.
    3. keep rpn_post_nms_topn proposals by their sorted scores from the highest
       to the lowest.
  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/ops/collect_and_distribute_fpn_rpn_proposals.py  # pylint: disable=line-too-long

  Args:
    scores_outputs: an OrderedDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderedDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4]
    all_anchors: an Anchors object that contains the all anchors.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    rpn_pre_nms_topn: a integer number of top scoring RPN proposals to keep
      before applying NMS. This is *per FPN level* (not total).
    rpn_post_nms_topn: a integer number of top scoring RPN proposals to keep
      after applying NMS. This is the total number of RPN proposals produced.
    rpn_nms_threshold: a float number between 0 and 1 as the NMS threshold
      used on RPN proposals.
    rpn_min_size: a integer number as the minimum proposal height and width as
      both need to be greater than this number. Note that this number is at
      origingal image scale; not scale used during training or inference).
  Returns:
    scores: a tensor with a shape of [batch_size, rpn_post_nms_topn, 1]
      representing the scores of the proposals.
    rois: a tensor with a shape of [batch_size, rpn_post_nms_topn, 4]
      representing the boxes of the proposals. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax].
  """
  with tf.name_scope('proposal'):
    levels = scores_outputs.keys()
    scores = []
    rois = []
    anchor_boxes = all_anchors.get_unpacked_boxes()
    for level in levels:
      # Expands the batch dimension for anchors as anchors do not have batch
      # dimension. Note that batch_size is invariant across levels.
      batch_size = scores_outputs[level].shape[0]
      anchor_boxes_batch = tf.cast(
          tf.tile(tf.expand_dims(anchor_boxes[level], axis=0),
                  [batch_size, 1, 1]),
          dtype=scores_outputs[level].dtype)
      scores_per_level, boxes_per_level = _proposal_op_per_level(
          scores_outputs[level], box_outputs[level], anchor_boxes_batch,
          image_info, rpn_pre_nms_topn, rpn_post_nms_topn, rpn_nms_threshold,
          rpn_min_size, level)
      scores.append(scores_per_level)
      rois.append(boxes_per_level)
    scores = tf.concat(scores, axis=1)
    rois = tf.concat(rois, axis=1)

    with tf.name_scope('post_nms_topk'):
      # Selects the top-k rois, k being rpn_post_nms_topn or the number of total
      # anchors after non-max suppression.
      post_nms_num_anchors = scores.shape[1]
      if post_nms_num_anchors < rpn_post_nms_topn:
        post_nms_topk_limit = post_nms_num_anchors
      else:
        post_nms_topk_limit = rpn_post_nms_topn

      top_k_scores, top_k_rois = box_utils.top_k(scores, k=post_nms_topk_limit,
                                                 tensors=[rois])
      top_k_rois = top_k_rois[0]
    top_k_scores = tf.stop_gradient(top_k_scores)
    top_k_rois = tf.stop_gradient(top_k_rois)
    return top_k_scores, top_k_rois


def rpn_net(features, min_level=2, max_level=6, num_anchors=3):
  """Region Proposal Network (RPN) for Mask-RCNN."""
  scores_outputs = {}
  box_outputs = {}
  with tf.variable_scope('rpn_net', reuse=tf.AUTO_REUSE):

    def shared_rpn_heads(features, num_anchors):
      """Shared RPN heads."""
      # TODO(chiachenc): check the channel depth of the first convoultion.
      features = tf.layers.conv2d(
          features,
          256,
          kernel_size=(3, 3),
          strides=(1, 1),
          activation=tf.nn.relu,
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='same',
          name='rpn')
      # Proposal classification scores
      scores = tf.layers.conv2d(
          features,
          num_anchors,
          kernel_size=(1, 1),
          strides=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='valid',
          name='rpn-class')
      # Proposal bbox regression deltas
      bboxes = tf.layers.conv2d(
          features,
          4 * num_anchors,
          kernel_size=(1, 1),
          strides=(1, 1),
          bias_initializer=tf.zeros_initializer(),
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          padding='valid',
          name='rpn-box')

      return scores, bboxes

    for level in range(min_level, max_level + 1):
      scores_output, box_output = shared_rpn_heads(features[level], num_anchors)
      scores_outputs[level] = scores_output
      box_outputs[level] = box_output

  return scores_outputs, box_outputs


def dense_with_relu(features, units, name):
  """Densely-connected layer with relu."""
  _, _, input_dims = features.get_shape().as_list()
  with tf.variable_scope(name):
    w_init = tf.compat.v1.keras.initializers.glorot_uniform()
    weights = tf.get_variable(
        name='weight',
        initializer=w_init,
        shape=[input_dims, units],
        trainable=True)
    weights = tf.cast(weights, dtype=features.dtype)
    b_init = tf.zeros_initializer()
    bias = tf.compat.v1.get_variable(
        name='bias', shape=[units], initializer=b_init, trainable=True)
    bias = tf.cast(bias, dtype=features.dtype)

    return tf.nn.relu(tf.einsum('bni,io->bno', features, weights) + bias)


def faster_rcnn_heads(features,
                      boxes,
                      num_classes=91,
                      mlp_head_dim=1024,
                      core_assignment=None,
                      num_partitions=1):
  """Box and class branches for the Mask-RCNN model.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/fast_rcnn_heads.py  # pylint: disable=line-too-long

  Args:
    features: A dictionary with key as pyramid level and value as features.
      The features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    num_classes: a integer for the number of classes.
    mlp_head_dim: a integer that is the hidden dimension in the fully-connected
      layers.
    core_assignment: A `int` to specify the core where the op is placed.
    num_partitions: A `int` indicates number of partiions per replica in SPMD.
  Returns:
    class_outputs: a tensor with a shape of
      [batch_size, num_rois, num_classes], representing the class predictions.
    box_outputs: a tensor with a shape of
      [batch_size, num_rois, num_classes * 4], representing the box predictions.
  """
  with tf.variable_scope('faster_rcnn_heads'):

    # Performs multi-level RoIAlign.
    roi_features = multilevel_crop_and_resize(features, boxes, output_size=7,
                                              core_assignment=core_assignment,
                                              num_partitions=num_partitions)
    with tf.device(core_assignment):
      # reshape inputs beofre FC.
      batch_size, num_rois, _, _, _ = roi_features.get_shape().as_list()
      if num_partitions is not None and num_partitions > 1:
        roi_features = xla_sharding.split(
            roi_features, 1, num_partitions, use_sharding_op=True)
      roi_features = tf.reshape(roi_features, [batch_size, num_rois, -1])

      net = dense_with_relu(roi_features, units=mlp_head_dim, name='fc6')
      net = dense_with_relu(net, units=mlp_head_dim, name='fc7')

      class_outputs = tf.layers.dense(
          net, num_classes,
          kernel_initializer=tf.random_normal_initializer(stddev=0.01),
          bias_initializer=tf.zeros_initializer(),
          name='class-predict')
      box_outputs = tf.layers.dense(
          net, num_classes * 4,
          kernel_initializer=tf.random_normal_initializer(stddev=0.001),
          bias_initializer=tf.zeros_initializer(),
          name='box-predict')
      return class_outputs, box_outputs


def mask_rcnn_heads(features,
                    fg_box_rois,
                    num_classes=91,
                    mrcnn_resolution=28,
                    core_assignment=None,
                    num_partitions=1):
  """Mask branch for the Mask-RCNN model.

  Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/modeling/mask_rcnn_heads.py  # pylint: disable=line-too-long

  Args:
    features: A dictionary with key as pyramid level and value as features.
      The features are in shape of [batch_size, height_l, width_l, num_filters].
    fg_box_rois: A 3-D Tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    num_classes: a integer for the number of classes.
    mrcnn_resolution: a integer that is the resolution of masks.
    core_assignment: A TensorFlow device to specify where the op is placed.
      `None` means no specification.
    num_partitions: A `int` indicates number of partiions per replica in SPMD.
  Returns:
    mask_outputs: a tensor with a shape of
      [batch_size, num_masks, mask_height, mask_width, num_classes],
      representing the mask predictions.
    fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
      representing the fg mask targets.
  Raises:
    ValueError: If boxes is not a rank-3 tensor or the last dimension of
      boxes is not 4.
  """

  def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
    """Returns the stddev of random normal initialization as MSRAFill."""
    # Reference: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/filler_op.h#L445-L463  # pylint: disable=line-too-long
    # For example, kernel size is (3, 3) and fan out is 256, stddev is 0.029.
    # stddev = (2/(3*3*256))^0.5 = 0.029
    return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

  if fg_box_rois.shape.ndims != 3:
    raise ValueError('fg_box_rois must be of rank 3.')
  if fg_box_rois.shape[2] != 4:
    raise ValueError(
        'fg_box_rois.shape[1] is {:d}, but must be divisible by 4.'.format(
            fg_box_rois.shape[1])
    )
  with tf.variable_scope('mask_rcnn_heads'):
    batch_size, num_masks, _ = fg_box_rois.get_shape().as_list()
    # Performs multi-level RoIAlign.
    features = multilevel_crop_and_resize(features, fg_box_rois, output_size=14,
                                          core_assignment=core_assignment,
                                          num_partitions=num_partitions)

    # Co-locates the following operations on the same core to avoid unnecessary
    # communication (only used in the model paralleism mode).
    with tf.device(core_assignment):
      if num_partitions is not None and num_partitions > 1:
        features = xla_sharding.split(
            features, 2, num_partitions, use_sharding_op=True)
      net = tf.reshape(
          features,
          [batch_size * num_masks, 14, 14, -1])

      # TODO(chiachenc): check what is MSRAFill initialization in the reference.
      for i in range(4):
        kernel_size = (3, 3)
        fan_out = 256
        init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
        net = tf.layers.conv2d(
            net,
            fan_out,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding='same',
            dilation_rate=(1, 1),
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
            bias_initializer=tf.zeros_initializer(),
            name='mask-conv-l%d' % i)

      kernel_size = (2, 2)
      fan_out = 256
      init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
      net = tf.layers.conv2d_transpose(
          net,
          fan_out,
          kernel_size=kernel_size,
          strides=(2, 2),
          padding='valid',
          activation=tf.nn.relu,
          kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
          bias_initializer=tf.zeros_initializer(),
          name='conv5-mask')

      kernel_size = (1, 1)
      fan_out = num_classes
      init_stddev = _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out)
      mask_outputs = tf.layers.conv2d(
          net,
          fan_out,
          kernel_size=kernel_size,
          strides=(1, 1),
          padding='valid',
          kernel_initializer=tf.random_normal_initializer(stddev=init_stddev),
          bias_initializer=tf.zeros_initializer(),
          name='mask_fcn_logits')
      if num_partitions is not None and num_partitions > 1:
        mask_outputs = xla_sharding.split(
            mask_outputs, 1, num_partitions, use_sharding_op=True)
      mask_outputs = tf.reshape(
          mask_outputs,
          [batch_size, num_masks, mrcnn_resolution, mrcnn_resolution, -1])

    return mask_outputs


def select_fg_for_masks(class_targets, box_targets, boxes,
                        proposal_to_label_map, max_num_fg=128):
  """Selects the fore ground objects for mask branch during training.

  Args:
    class_targets: a tensor of shape [batch_size, num_boxes]  representing the
      class label for each box.
    box_targets: a tensor with a shape of [batch_size, num_boxes, 4]. The tensor
      contains the ground truth pixel coordinates of the scaled images for each
      roi.
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    proposal_to_label_map: a tensor with a shape of [batch_size, num_boxes].
      This tensor keeps the mapping between proposal to labels.
      proposal_to_label_map[i] means the index of the ground truth instance for
      the i-th proposal.
    max_num_fg: a integer represents the number of masks per image.
  Returns:
    class_targets, boxes, proposal_to_label_map, box_targets that have
    foreground objects.
  """
  with tf.name_scope('select_fg_for_masks'):
    # Masks are for positive (fg) objects only. Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/roi_data/mask_rcnn.py  # pylint: disable=line-too-long
    batch_size = boxes.shape[0]
    _, fg_indices = tf.nn.top_k(
        tf.to_float(tf.greater(class_targets, 0)), k=max_num_fg)
    # Contructs indices for gather.
    indices = tf.reshape(
        fg_indices + tf.expand_dims(
            tf.range(batch_size) * tf.shape(class_targets)[1], 1), [-1])

    fg_class_targets = tf.reshape(
        tf.gather(tf.reshape(class_targets, [-1, 1]), indices),
        [batch_size, -1])
    fg_box_targets = tf.reshape(
        tf.gather(tf.reshape(box_targets, [-1, 4]), indices),
        [batch_size, -1, 4])
    fg_box_rois = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), indices), [batch_size, -1, 4])
    fg_proposal_to_label_map = tf.reshape(
        tf.gather(tf.reshape(proposal_to_label_map, [-1, 1]), indices),
        [batch_size, -1])

  return (fg_class_targets, fg_box_targets, fg_box_rois,
          fg_proposal_to_label_map)


def mask_post_processing(class_indices, num_classes, mask_outputs):
  """Performs post-processing for masks.

  This function uses `classes_indices` to select the mask predictions from
  `mask_outputs`. In PREDICT mode, the `class_indices` is from Faster-RCNN
  heads; in TRAIN mode, the `class_indices` is from foreground class targets.
  The algorithm is based on gather but implemented in one-hot plus einsum.

  The gather-based algorithm:
  ```
    batch_size, num_masks = class_indices.get_shape().as_list()
    mask_outputs = tf.transpose(mask_outputs, [0, 1, 4, 2, 3])
    # Contructs indices for gather.
    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, num_masks])
    mask_indices = tf.tile(
        tf.expand_dims(tf.range(num_masks), axis=0), [batch_size, 1])
    gather_indices = tf.stack(
        [batch_indices, mask_indices, class_indices], axis=2)
    mask_outputs = tf.gather_nd(mask_outputs, gather_indices)
  ```

  Args:
    class_indices: A tensor with a shape of [batch_size, max_num_fg]
      representing class indices of each target.
    num_classes: A `int` that is the number of classes.
    mask_outputs: A tensor with a shape of [batch_size, max_num_fg,
      mrcnn_resolution, mrcnn_resolution, num_classes] representing class-aware
      mask predictions of each object.
  Returns:
      A tensor with a shape of [batch_size, max_num_fg, mrcnn_resolution,
      mrcnn_resolution] representing the mask prediction of each object.
  """
  # Performs post-processing for mask outputs.
  with tf.name_scope('masks_post_processing'):
    one_hot = tf.one_hot(class_indices, depth=num_classes,
                         dtype=mask_outputs.dtype)
    mask_outputs = tf.einsum('bnhwf,bnf->bnhw', mask_outputs, one_hot)
    return mask_outputs


def faster_rcnn(features,
                rpn_score_outputs,
                rpn_box_outputs,
                all_anchors,
                image_info,
                is_training,
                params,
                labels=None):
  """Faster-RCNN classification and regression model.

  This is the box and the class part of the model (Faster-RCNN). It generates
  class_outputs, box_outputs, and box_rois. In addition, it generates training
  targets in TRAIN mode.

  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    rpn_score_outputs: An OrderedDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    rpn_box_outputs: An OrderedDict with keys representing levels and values
      representing box regression targets in [batch_size, height, width,
      num_anchors * 4]
    all_anchors: An Anchors object that contains the all anchors.
    image_info: a tensor of shape [batch_size, 5] where the three columns
      encode the input image's [height, width, scale,
      original_height, original_width]. Height and width are for
      the input to the network, not the original image; scale is the scale
      factor used to scale the network input size to the original image size.
      See dataloader.DetectionInputProcessor for details. The last two are
      original height and width. See dataloader.DetectionInputProcessor for
      details.
    is_training: whether this is training or predcit.
    params: A dictionary defines hyperparameters of model. The default
      settings are in default_hparams() function in mask_rcnn_params.py.
    labels: (for training only) The input labels in a dictionary. The labels
      include groundtruth boxes and classes.
  Returns:
    class_outputs: A tensor with a shape of [batch_size, batch_size_per_im,
      num_classes] representing the class prediction of each object.
    box_outputs: A tensor with a shape of [batch_size, batch_size_per_im,
      num_classes * 4] representing the box prediction of each object.
    box_rois: A tensor with a shape of [batch_size, batch_size_per_im, 4]
      representing the proposal boxes. The boxes are in normalized
      coordinates with a form of [ymin, xmin, ymax, xmax].
    The following is for training only.
    class_targets: A tensor with a shape of [batch_size, batch_size_per_im,
      num_classes] representing the classification target of each object.
    box_targets: A tensor with a shape of [batch_size, batch_size_per_im,
      num_classes * 4] representing the box regression target of each object.
    proposal_to_label_map: A tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels. proposal_to_label_map[i]
      means the index of the ground truth instance for the i-th proposal.
  Raises:
    ValueError: labels must be present in TRAIN and absent in PREDICT.
  """
  if is_training and labels is None:
    raise ValueError('labels must be provided in TRAIN mode.')
  if (not is_training) and labels is not None:
    raise ValueError('labels should be omitted in PREDICT mode.')

  # Uses different NMS top-k parameters in different modes.
  if is_training:
    rpn_pre_nms_topn = params['rpn_pre_nms_topn']
    rpn_post_nms_topn = params['rpn_post_nms_topn']
  else:
    rpn_pre_nms_topn = params['test_rpn_pre_nms_topn']
    rpn_post_nms_topn = params['test_rpn_post_nms_topn']
  _, box_rois = proposal_op(rpn_score_outputs, rpn_box_outputs, all_anchors,
                            image_info, rpn_pre_nms_topn,
                            rpn_post_nms_topn, params['rpn_nms_threshold'],
                            params['rpn_min_size'])
  box_rois = tf.to_float(box_rois)

  if is_training:
    (box_targets, class_targets, box_rois,
     proposal_to_label_map) = proposal_label_op(
         box_rois, labels['gt_boxes'], labels['gt_classes'],
         image_info, batch_size_per_im=params['batch_size_per_im'],
         fg_fraction=params['fg_fraction'], fg_thresh=params['fg_thresh'],
         bg_thresh_hi=params['bg_thresh_hi'],
         bg_thresh_lo=params['bg_thresh_lo'])

  class_outputs, box_outputs = faster_rcnn_heads(
      features,
      box_rois,
      num_classes=params['num_classes'],
      mlp_head_dim=params['fast_rcnn_mlp_head_dim'],
      core_assignment=core_assignment_utils.get_core_assignment(
          core_assignment_utils.CORE_2, params['num_cores_per_replica'],
          params['use_spmd']),
      num_partitions=params['num_cores_per_replica']
      if params['use_spmd'] else 1)

  if is_training:
    return (class_outputs, box_outputs, box_rois, class_targets, box_targets,
            proposal_to_label_map)
  else:
    return (class_outputs, box_outputs, box_rois)


def mask_rcnn(features,
              is_training,
              params,
              labels=None,
              class_targets=None,
              box_targets=None,
              box_rois=None,
              proposal_to_label_map=None,
              detections=None):
  """Mask-RCNN mask model.

  This is the mask part of the model (Mask-RCNN), which generates mask outputs.
  In addition, it also generates mask targets and corresponding class targets
  during training (see return docstring). The mask outputs are selected by the
  class prediction (in PREDICT) or th class target (in TRAIN).

  Note that this mask part considers only foreground objects.

  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    is_training: Whethere this is training or predict.
    params: A dictionary defines hyperparameters of model. The default settings
      are in default_hparams() function in mask_rcnn_params.py.
    labels: The input labels in a dictionary. The labels include groundtruth
      masks, boxes and classes.
    class_targets: A tensor with a shape of [batch_size, batch_size_per_im,
      num_classes] representing the classification target of each object.
    box_targets: A tensor with a shape of [batch_size, batch_size_per_im,
      num_classes] representing the classification target of each object.
    box_rois: A tensor with a shape of [batch_size, batch_size_per_im, 4]
      representing the proposal boxes.
    proposal_to_label_map: A tensor with a shape of [batch_size, K]. This tensor
      keeps the mapping between proposal to labels.
    detections: A tensor with a shape of [batch_size, num_detections, 7],
      representing the class and box predictions in PREDICT mode.

  Returns:
    mask_outputs: A tensor with a shape of [batch_size, max_num_fg,
      mrcnn_resolution, mrcnn_resolution] representing the mask prediction of
      each object.
    The following is for training only.
    mask_targets: A tensor with a shape of [batch_size, max_num_fg,
      mrcnn_resolution, mrcnn_resolution] representing the mask target of each
      object.
    class_targets: A tensor with a shape of [batch_size, max_num_fg,
      num_classes * 4] representing the classification target of each object.
  """
  if is_training:
    (class_targets, box_targets, box_rois,
     proposal_to_label_map) = select_fg_for_masks(
         class_targets, box_targets, box_rois, proposal_to_label_map,
         max_num_fg=int(params['batch_size_per_im'] * params['fg_fraction']))
    device = core_assignment_utils.get_core_assignment(
        core_assignment_utils.CORE_0, params['num_cores_per_replica'],
        params['use_spmd'])
    with tf.device(device):
      mask_targets = get_mask_targets(
          box_rois,
          proposal_to_label_map,
          box_targets,
          labels['cropped_gt_masks'],
          params['mrcnn_resolution'],
          num_partitions=params['num_cores_per_replica']
          if params['use_spmd'] else 1)
    mask_outputs = mask_rcnn_heads(
        features,
        box_rois,
        num_classes=params['num_classes'],
        mrcnn_resolution=params['mrcnn_resolution'],
        core_assignment=core_assignment_utils.get_core_assignment(
            core_assignment_utils.CORE_1, params['num_cores_per_replica'],
            params['use_spmd']),
        num_partitions=params['num_cores_per_replica']
        if params['use_spmd'] else 1)
    class_indices = tf.to_int32(class_targets)
  else:
    box_rois = detections[:, :, 1:5]
    mask_outputs = mask_rcnn_heads(
        features,
        box_rois,
        num_classes=params['num_classes'],
        mrcnn_resolution=params['mrcnn_resolution'],
        core_assignment=core_assignment_utils.get_core_assignment(
            core_assignment_utils.CORE_1, params['num_cores_per_replica'],
            params['use_spmd']),
        num_partitions=params['num_cores_per_replica']
        if params['use_spmd'] else 1)
    class_indices = tf.to_int32(detections[:, :, 6])

  device = core_assignment_utils.get_core_assignment(
      core_assignment_utils.CORE_1, params['num_cores_per_replica'],
      params['use_spmd'])
  with tf.device(device):
    mask_outputs = mask_post_processing(class_indices, params['num_classes'],
                                        mask_outputs)

  if is_training:
    return mask_outputs, class_targets, mask_targets
  else:
    return mask_outputs


def get_mask_targets(fg_boxes,
                     fg_proposal_to_label_map,
                     fg_box_targets,
                     mask_gt_labels,
                     output_size=28,
                     num_partitions=1):
  """Crop and resize on ground truth masks.

  This is a three stage Gather function.
  1. The first Gather gathers feature for each box from `mask_gt_labels`,
     indices is `fg_proposal_to_label_map`;
  2. The second Gather gathers y_axis grid pixels from per box features;
  3. The third Gather gathers x_axis grid pixels from per box features.

  Blinear interpolation is done during the last two gathers:
            f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                                  [f10, f11]]
            [[f00, f01],
             [f10, f11]] = tf.einsum(tf.einsum(features, y_one_hot), x_one_hot)
            where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

  Gathers are replaced with einsums for performance.

  Args:
    fg_boxes: A 3-D tensor of shape [batch_size, num_masks, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    fg_proposal_to_label_map: A tensor of shape [batch_size, num_masks].
    fg_box_targets: a float tensor representing the box label for each box
      with a shape of [batch_size, num_masks, 4].
    mask_gt_labels: A tensor with a shape of [batch_size, M, H+4, W+4, 4]. M is
      NUM_MAX_INSTANCES (i.e., 100 in this implementation) in each image, while
      H and W are ground truth mask size. The `+4` comes from padding of two
      zeros in both directions of height and width dimension.
    output_size: A scalar to indicate the output crop size.
    num_partitions: A `int` indicates number of partiions per replica in SPMD.


  Returns:
    A 4-D tensor representing ground truth masks with a shape of [batch_size,
    num_boxes, output_size, output_size].
  """
  with tf.name_scope('get_mask_targets'):
    (batch_size, num_instances, max_feature_height,
     max_feature_width) = mask_gt_labels.get_shape().as_list()
    _, num_masks = fg_proposal_to_label_map.get_shape().as_list()

    # proposal_to_label_map might have a -1 paddings.
    levels = tf.maximum(fg_proposal_to_label_map, 0)
    levels_one_hot = tf.one_hot(tf.cast(levels, tf.int32), num_instances)
    # 1st Gather: gather features for each box.
    # shape of [batch_size, num_masks, height, width]
    features_per_box = tf.einsum('bihw,bmi->bmhw', mask_gt_labels,
                                 levels_one_hot)

    # Projects box location and sizes to corresponding cropped ground truth
    # mask coordinates.
    if num_partitions is not None and num_partitions > 1:
      fg_boxes = xla_sharding.split(
          fg_boxes, 1, num_partitions, use_sharding_op=True)
      fg_box_targets = xla_sharding.split(
          fg_box_targets, 1, num_partitions, use_sharding_op=True)
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=fg_boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=fg_box_targets, num_or_size_splits=4, axis=2)
    valid_feature_width = max_feature_width - 4
    valid_feature_height = max_feature_height - 4
    y_transform = (bb_y_min - gt_y_min) * valid_feature_height / (
        gt_y_max - gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * valid_feature_width / (
        gt_x_max - gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * valid_feature_height / (
        gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * valid_feature_width / (
        gt_x_max - gt_x_min + _EPSILON)

    # Compute y and x coordinate indices.
    box_grid_y = []
    box_grid_x = []
    for i in range(output_size):
      box_grid_y.append(y_transform + (0.5 + i) * h_transform / output_size)
      box_grid_x.append(x_transform + (0.5 + i) * w_transform / output_size)
    box_grid_y = tf.stack(box_grid_y, axis=2)
    box_grid_x = tf.stack(box_grid_x, axis=2)

    box_grid_y0 = tf.floor(box_grid_y)
    box_grid_x0 = tf.floor(box_grid_x)

    # Compute indices for gather operation.
    box_gridx0x1 = tf.stack([box_grid_x0, box_grid_x0 + 1], axis=3)
    box_gridy0y1 = tf.stack([box_grid_y0, box_grid_y0 + 1], axis=3)

    # Check boundary.
    box_gridx0x1 = tf.minimum(
        tf.to_float(max_feature_width - 1), tf.maximum(0., box_gridx0x1))
    box_gridy0y1 = tf.minimum(
        tf.to_float(max_feature_height - 1), tf.maximum(0., box_gridy0y1))

    if num_partitions is not None and num_partitions > 1:
      box_gridy0y1 = xla_sharding.split(
          box_gridy0y1, 1, num_partitions, use_sharding_op=True)
      box_gridx0x1 = xla_sharding.split(
          box_gridx0x1, 1, num_partitions, use_sharding_op=True)

    # shape is [batch_size, num_masks, output_size, 2]
    x_indices = tf.cast(
        tf.reshape(box_gridx0x1, [batch_size, num_masks, output_size, 2]),
        dtype=tf.int32)
    y_indices = tf.cast(
        tf.reshape(box_gridy0y1, [batch_size, num_masks, output_size, 2]),
        dtype=tf.int32)

    # TODO(wangtao): convert the common logic with multilevel_crop_and_resize
    # to a seperate function.

    # The output can be computed by bilinear interpolation of four neighboring
    # points f0, f1, f2, and f3.
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    #
    # [[f00, f01],
    # [f10, f11]] = tf.einsum(tf.einsum(features, y_one_hot), x_one_hot)
    #

    # Shape is [batch_size, num_masks, output_size]
    ly = box_grid_y - box_grid_y0
    lx = box_grid_x - box_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx

    if num_partitions is not None and num_partitions > 1:
      hy = xla_sharding.split(hy, 1, num_partitions, use_sharding_op=True)
      ly = xla_sharding.split(ly, 1, num_partitions, use_sharding_op=True)
      hx = xla_sharding.split(hx, 1, num_partitions, use_sharding_op=True)
      lx = xla_sharding.split(lx, 1, num_partitions, use_sharding_op=True)
    # shape is [batch_size, num_masks, output_size, 2, 1]
    # divide-by-sampling_ratio is for averaging.
    kernel_y = tf.reshape(
        tf.stack([hy, ly], axis=3), [batch_size, num_masks, output_size, 2, 1])
    kernel_x = tf.reshape(
        tf.stack([hx, lx], axis=3), [batch_size, num_masks, output_size, 2, 1])

    # shape is [batch_size, num_masks, output_size, 2, height]
    grid_y_one_hot = tf.one_hot(
        tf.cast(y_indices, tf.int32), max_feature_height)
    # shape is [batch_size, num_masks, output_size, 2, width]
    grid_x_one_hot = tf.one_hot(tf.cast(x_indices, tf.int32), max_feature_width)

    # shape is [batch_size, num_masks, output_size, height]
    grid_y_weight = tf.reduce_sum(
        tf.multiply(grid_y_one_hot, kernel_y), axis=-2)
    # shape is [batch_size, num_masks, output_size, width]
    grid_x_weight = tf.reduce_sum(
        tf.multiply(grid_x_one_hot, kernel_x), axis=-2)

    if num_partitions is not None and num_partitions > 1:
      grid_y_weight = xla_sharding.split(
          grid_y_weight, 1, num_partitions, use_sharding_op=True)
      grid_x_weight = xla_sharding.split(
          grid_x_weight, 1, num_partitions, use_sharding_op=True)

    # 2nd Gather: gather for y_axis.
    # shape is [batch_size, num_masks, output_size, width]
    features_per_box = tf.einsum('bmhw,bmoh->bmow', features_per_box,
                                 grid_y_weight)
    # 3rd Gather: gather for x_axis.
    # shape is [batch_size, num_masks, output_size, output_size]
    features_per_box = tf.einsum('bmhw,bmow->bmho', features_per_box,
                                 grid_x_weight)

    # Masks are binary outputs.
    features_per_box = tf.cast(
        tf.greater_equal(features_per_box, 0.5), dtype=features_per_box.dtype)

    # mask_targets depend on box RoIs, which have gradients. This
    # stop_gradient prevents the flow of gradient to box RoIs.
    features_per_box = tf.stop_gradient(features_per_box)
  return features_per_box


def multilevel_crop_and_resize_v1(features,
                                  boxes,
                                  output_size=7,
                                  core_assignment=None):
  """Crop and resize on multilevel feature pyramid.

    Following the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
    figure 3 for reference), we want to sample pixel level feature information
    from our feature map at the box boundaries.  For each feature map, we select
    an (output_size, output_size) set of pixels corresponding to our box
    location, and then use bilinear interpolation to select the feature value
    for each pixel.

    For performance, we perform the gather and interpolation on all layers as a
    single operation. This is op the multi-level features are first stacked and
    gathered into [2*output_size, 2*output_size] feature points. Then bilinear
    interpolation is performed on the gathered feature points to generate
    [output_size, output_size] RoIAlign feature map.

    Here is the step-by-step algorithm:
    1. Stack multi-level features into a Tensor of shape [batch_size,
       total_feature_size, num_filters], where
       total_feature_size = sum_for_all_levels(level_width * level_height).
    2. Compute four neighboring x and y grid indicies for each output point
       for each boxes. Each box will be represented as
       [output_size, output_size] output points,
       thus [output_size*2, output_size*2] neighboring points are needed.
       The neighboring indices are the floor and ceiling of the mapping
       coordinates on x or y axis. x gird indices and y grid indices are
       computed seperately.
    3. Compute indices to gather the four neighboring indices from feature map.
    4. Gather the Tensor containing multi-level features.
       The tensor with a shape
       [batch_size*num_boxes*output_size*2*output_size*2, num_filters]
       is reshaped to
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters].
    3. Compute the interpolation kernel of shape
       [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
       can be seen as stacking 2x2 interpolation kernels for all vertices in the
       output grid.
    4. Element-wise multiply the gathered features and interpolation kernel.
       Then apply 2x2 average pooling to reduce spatial dimension to
       output_size.

  Args:
    features: A dictionary with key as pyramid level and value as features.
      The features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
      represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.
    core_assignment: A TensorFlow device to specify where the op is placed.
      `None` means no specification.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """
  with tf.name_scope('multilevel_crop_and_resize'):
    levels = features.keys()
    min_level = min(levels)
    max_level = max(levels)
    (batch_size, max_feature_height,
     max_feature_width, num_filters) = features[min_level].get_shape().as_list()
    _, num_boxes, _ = boxes.get_shape().as_list()
    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    features_all = []
    feature_heights = []
    feature_widths = []
    for level in range(min_level, max_level + 1):
      shape = features[level].get_shape().as_list()
      feature_heights.append(shape[1])
      feature_widths.append(shape[2])
      features_all.append(
          tf.reshape(features[level], [batch_size, -1, num_filters]))
    features_r2 = tf.reshape(tf.concat(features_all, 1), [-1, num_filters])

    with tf.device(core_assignment):
      level_dim_sizes = [
          feature_widths[i] * feature_heights[i]
          for i in range(len(feature_widths))
      ]
      level_dim_offsets = [0]
      for i in range(len(feature_widths) - 1):
        level_dim_offsets.append(level_dim_offsets[i] + level_dim_sizes[i])
      batch_dim_size = level_dim_offsets[-1] + level_dim_sizes[-1]
      level_dim_offsets = tf.constant(level_dim_offsets, tf.int32)
      height_dim_sizes = tf.constant(feature_widths, tf.int32)

      # Assign boxes to the right level.
      box_width = boxes[:, :, 3] - boxes[:, :, 1]
      box_height = boxes[:, :, 2] - boxes[:, :, 0]
      areas_sqrt = tf.sqrt(box_height * box_width)
      levels = tf.cast(tf.floordiv(tf.log(tf.div(areas_sqrt, 224.0)),
                                   tf.log(2.0)) + 4.0, dtype=tf.int32)
      # Map levels between [min_level, max_level].
      levels = tf.minimum(max_level, tf.maximum(levels, min_level))

      # Project box location and sizes to corresponding feature levels.
      scale_to_level = tf.cast(
          tf.pow(tf.constant(2.0), tf.cast(levels, tf.float32)),
          dtype=boxes.dtype)
      boxes /= tf.expand_dims(scale_to_level, axis=2)
      box_width /= scale_to_level
      box_height /= scale_to_level

      # Map levels to [0, max_level-min_level].
      levels -= min_level

      # Compute y and x coordinate indices.
      box_grid_x = []
      box_grid_y = []
      for i in range(output_size):
        box_grid_x.append(boxes[:, :, 1] + (i + 0.5) * box_width / output_size)
        box_grid_y.append(boxes[:, :, 0] + (i + 0.5) * box_height / output_size)
      box_grid_x = tf.stack(box_grid_x, axis=2)
      box_grid_y = tf.stack(box_grid_y, axis=2)

      box_grid_y0 = tf.floor(box_grid_y)
      box_grid_x0 = tf.floor(box_grid_x)

      # Compute indices for gather operation.
      box_grid_x0 = tf.maximum(0., box_grid_x0)
      box_grid_y0 = tf.maximum(0., box_grid_y0)
      boundary_width = tf.cast(
          tf.expand_dims([[tf.cast(max_feature_width, tf.float32)]] /
                         tf.pow([[2.0]], tf.cast(levels, tf.float32)) - 1, 2),
          box_grid_x0.dtype)
      box_gridx0x1 = tf.stack([
          tf.minimum(box_grid_x0, boundary_width),
          tf.minimum(box_grid_x0 + 1, boundary_width)
      ],
                              axis=3)
      boundary_height = tf.cast(
          tf.expand_dims([[tf.cast(max_feature_height, tf.float32)]] /
                         tf.pow([[2.0]], tf.cast(levels, tf.float32)) - 1, 2),
          box_grid_y0.dtype)
      box_gridy0y1 = tf.stack([
          tf.minimum(box_grid_y0, boundary_height),
          tf.minimum(box_grid_y0 + 1, boundary_height)
      ],
                              axis=3)

      x_indices = tf.cast(
          tf.reshape(box_gridx0x1,
                     [batch_size, num_boxes, output_size * 2]), dtype=tf.int32)
      y_indices = tf.cast(
          tf.reshape(box_gridy0y1,
                     [batch_size, num_boxes, output_size * 2]), dtype=tf.int32)

      indices = tf.reshape(
          tf.tile(
              tf.reshape(
                  tf.range(batch_size) * batch_dim_size, [batch_size, 1, 1, 1]),
              [1, num_boxes, output_size * 2, output_size * 2]) + tf.tile(
                  tf.reshape(
                      tf.gather(level_dim_offsets, levels),
                      [batch_size, num_boxes, 1, 1]),
                  [1, 1, output_size * 2, output_size * 2]) +
          tf.tile(
              tf.reshape(
                  y_indices *
                  tf.expand_dims(tf.gather(height_dim_sizes, levels), -1),
                  [batch_size, num_boxes, output_size * 2, 1]),
              [1, 1, 1, output_size * 2]) +
          tf.tile(
              tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2
                                    ]), [1, 1, output_size * 2, 1]), [-1])

      features_per_box = tf.reshape(
          tf.gather(features_r2, indices),
          [batch_size, num_boxes, output_size * 2, output_size * 2,
           num_filters])

      # The RoIAlign feature f can be computed by bilinear interpolation of four
      # neighboring feature points f0, f1, f2, and f3.
      # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
      #                       [f10, f11]]
      # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
      # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
      ly = box_grid_y - box_grid_y0
      lx = box_grid_x - box_grid_x0
      hy = 1.0 - ly
      hx = 1.0 - lx
      kernel_x = tf.reshape(tf.stack([hx, lx], axis=3),
                            [batch_size, num_boxes, 1, output_size*2])
      kernel_y = tf.reshape(tf.stack([hy, ly], axis=3),
                            [batch_size, num_boxes, output_size*2, 1])
      # Use implicit broadcast to generate the interpolation kernel. The
      # multiplier `4` is for avg pooling.
      interpolation_kernel = kernel_y * kernel_x * 4

      # Interpolate the gathered features with computed interpolation kernels.
      features_per_box *= tf.cast(
          tf.expand_dims(interpolation_kernel, axis=4),
          dtype=features_per_box.dtype)
      features_per_box = tf.reshape(
          features_per_box,
          [batch_size * num_boxes, output_size*2, output_size*2, num_filters])
      features_per_box = tf.nn.avg_pool(
          features_per_box, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
      features_per_box = tf.reshape(
          features_per_box,
          [batch_size, num_boxes, output_size, output_size, num_filters])

  return features_per_box


def multilevel_crop_and_resize(features,
                               boxes,
                               output_size=7,
                               sampling_ratio=2,
                               core_assignment=None,
                               num_partitions=1):
  """Crop and resize on multilevel feature pyramid.

    Following the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
    figure 3 for reference), we want to sample pixel level feature information
    from our feature map at the box boundaries.  For each feature map, we select
    an (output_size, output_size) set of pixels corresponding to our box
    location, and then use bilinear interpolation to select the feature value
    for each pixel.


    Here is the step-by-step algorithm:
    1. Compute sampling points and their four neighbors for each output points.
       Each box is mapped to [output_size, output_size] points.
       Each output point is averaged among #sampling_raitio^2 points.
       Each sampling point is computed using bilinear
       interpolation of its four neighboring points on the feature map.
    2. Gather output points seperately for each level. Gather and computation of
       output points are done for the boxes mapped to this level only.
       2.1. Compute indices of four neighboring point of each sampling
            point for x and y seperately of shape
            [batch_size, num_boxes, output_size, sampling_ratio * 2].
       2.2. Compute the interpolation kernel for axis x and y seperately of
            shape [batch_size, num_boxes, output_size, sampling_ratio * 2, 1].
       2.3. The features are colleced into a
            [batch_size, num_boxes, output_size, output_size, num_filters]
            Tensor.
            Instead of a one-step algorithm, a two-step approach is used.
            That is, first, an intermediate output is stored with a shape of
            [batch_size, num_boxes, output_size, width, num_filters];
            second, the final output is produced with a shape of
            [batch_size, num_boxes, output_size, output_size, num_filters].

            Blinear interpolation is done during the two step gather:
            f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                                  [f10, f11]]
            [[f00, f01],
             [f10, f11]] = tf.einsum(tf.einsum(features, y_one_hot), x_one_hot)
            where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

            Note:
              a. Use one_hot with einsum to replace gather;
              b. Bilinear interpolation and averaging of
                 multiple sampling points are fused into the one_hot vector.
  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row represents
      a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.
    sampling_ratio: A integer to indicate the sampling ratio.
    core_assignment: A TensorFlow device to specify where the op is placed.
      `None` means no specification.
    num_partitions: A `int` indicates number of partiions per replica in SPMD.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """
  with tf.name_scope('multilevel_crop_and_resize'):
    levels = features.keys()
    min_level = min(levels)
    max_level = max(levels)
    (batch_size, _, _, num_filters) = features[min_level].get_shape().as_list()
    if num_partitions is None:
      num_partitions = 1
    if num_partitions > 1:
      boxes = xla_sharding.split(boxes, 1, num_partitions, use_sharding_op=True)
    _, num_boxes, _ = boxes.get_shape().as_list()

    with tf.device(core_assignment):
      # Assign boxes to the right level.
      box_height = boxes[:, :, 2] - boxes[:, :, 0]
      box_width = boxes[:, :, 3] - boxes[:, :, 1]
      areas_sqrt = tf.sqrt(box_height * box_width)
      levels = tf.cast(
          tf.floordiv(tf.log(tf.div(areas_sqrt, 224.0)), tf.log(2.0)) + 4.0,
          dtype=tf.int32)
      # Map levels between [min_level, max_level].
      levels = tf.minimum(max_level, tf.maximum(levels, min_level))

      # Project box location and sizes to corresponding feature levels.
      scale_to_level = tf.cast(
          tf.pow(tf.constant(2.0), tf.cast(levels, tf.float32)),
          dtype=boxes.dtype)
      boxes /= tf.expand_dims(scale_to_level, axis=2)
      box_height /= scale_to_level
      box_width /= scale_to_level

      # Reference: http://shortn/_MWbs7NSWsY
      bin_size_h = box_height / output_size
      bin_size_w = box_width / output_size

      sampling_ratio = max(sampling_ratio, 1)
      # Compute y and x coordinates.
      box_grid_y = []
      box_grid_x = []
      for i in range(output_size):
        for j in range(sampling_ratio):
          y = (
              boxes[:, :, 0] + i * bin_size_h +
              (j + 0.5) * bin_size_h / sampling_ratio)
          x = (
              boxes[:, :, 1] + i * bin_size_w +
              (j + 0.5) * bin_size_w / sampling_ratio)

          box_grid_y.append(y)
          box_grid_x.append(x)
      # shape is [batch_size, num_boxes, output_size * sampling_ratio]
      box_grid_y = tf.stack(box_grid_y, axis=2)
      box_grid_x = tf.stack(box_grid_x, axis=2)

      def two_step_gather_per_level(features_level, mask):
        """Performs two-step gather using einsum for every level of features."""
        (_, feature_height, feature_width,
         _) = features_level.get_shape().as_list()

        # Generate one-hot vector for einsum.
        box_grid_y0 = tf.floor(box_grid_y)
        box_grid_x0 = tf.floor(box_grid_x)

        box_grid_y0 = tf.maximum(0., box_grid_y0)
        box_grid_x0 = tf.maximum(0., box_grid_x0)
        box_gridy0y1 = tf.stack([
            tf.minimum(box_grid_y0, feature_height),
            tf.minimum(box_grid_y0 + 1, feature_height)
        ],
                                axis=3)
        box_gridx0x1 = tf.stack([
            tf.minimum(box_grid_x0, feature_width),
            tf.minimum(box_grid_x0 + 1, feature_width)
        ],
                                axis=3)
        box_gridy0y1 = tf.reshape(
            box_gridy0y1,
            [batch_size, num_boxes, output_size, sampling_ratio * 2])
        box_gridx0x1 = tf.reshape(
            box_gridx0x1,
            [batch_size, num_boxes, output_size, sampling_ratio * 2])

        # The RoIAlign feature f can be computed by bilinear interpolation of
        # four neighboring feature points f00, f01, f10, and f11.
        # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
        #                       [f10, f11]]
        # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
        # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
        ly = box_grid_y - box_grid_y0
        lx = box_grid_x - box_grid_x0
        hy = 1.0 - ly
        hx = 1.0 - lx

        if num_partitions > 1:
          hy = xla_sharding.split(hy, 1, num_partitions, use_sharding_op=True)
          ly = xla_sharding.split(ly, 1, num_partitions, use_sharding_op=True)
          hx = xla_sharding.split(hx, 1, num_partitions, use_sharding_op=True)
          lx = xla_sharding.split(lx, 1, num_partitions, use_sharding_op=True)
        # shape is [batch_size, num_boxes, output_size, sampling_ratio * 2, 1]
        # divide-by-sampling_ratio is for averaging.
        kernel_y = tf.reshape(
            tf.stack([hy, ly], axis=3),
            [batch_size, num_boxes, output_size, sampling_ratio * 2, 1
            ]) / sampling_ratio
        kernel_x = tf.reshape(
            tf.stack([hx, lx], axis=3),
            [batch_size, num_boxes, output_size, sampling_ratio * 2, 1
            ]) / sampling_ratio

        # shape is:
        # [batch_size, num_boxes, output_size, sampling_ratio * 2, spatial_size]
        box_grid_y_one_hot = tf.one_hot(
            tf.cast(box_gridy0y1, tf.int32), feature_height)
        box_grid_x_one_hot = tf.one_hot(
            tf.cast(box_gridx0x1, tf.int32), feature_width)

        # # shape is [batch_size, num_boxes, output_size, spatial_size]
        box_grid_y_weight = tf.reduce_sum(
            tf.multiply(box_grid_y_one_hot, kernel_y), axis=-2)
        box_grid_x_weight = tf.reduce_sum(
            tf.multiply(box_grid_x_one_hot, kernel_x), axis=-2)

        if num_partitions > 1:
          box_grid_y_weight = xla_sharding.split(
              box_grid_y_weight, 1, num_partitions, use_sharding_op=True)
          box_grid_x_weight = xla_sharding.split(
              box_grid_x_weight, 1, num_partitions, use_sharding_op=True)
        # shape is [batch_size, num_boxes, output_size, width, feature]
        y_outputs = tf.einsum(
            'bhwf,bnyh->bnywf', features_level,
            tf.cast(box_grid_y_weight, dtype=features_level.dtype))

        # shape is [batch_size, num_boxes, output_size, output_size, feature]
        x_outputs = tf.einsum(
            'bnywf,bnxw->bnyxf', y_outputs,
            tf.cast(box_grid_x_weight, dtype=features_level.dtype))

        outputs = tf.where(
            tf.equal(mask, tf.zeros_like(mask)), tf.zeros_like(x_outputs),
            x_outputs)
        return outputs

      features_per_box = tf.zeros(
          [batch_size, num_boxes, output_size, output_size, num_filters],
          dtype=features[min_level].dtype)
      for level in range(min_level, max_level + 1):
        level_equal = tf.equal(levels, level)
        if num_partitions > 1:
          level_equal = xla_sharding.split(
              level_equal, 1, num_partitions, use_sharding_op=True)
        mask = tf.tile(
            tf.reshape(level_equal, [batch_size, num_boxes, 1, 1, 1]),
            [1, 1, output_size, output_size, num_filters])
        if num_partitions > 1:
          mask = xla_sharding.split(
              mask, 1, num_partitions, use_sharding_op=True)

        features_per_box += two_step_gather_per_level(features[level], mask)

      if num_partitions > 1:
        features_per_box = xla_sharding.split(
            features_per_box, 1, num_partitions, use_sharding_op=True)

  return features_per_box
