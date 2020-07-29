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
"""Model defination for the SSD Model.

Defines model_fn of SSD for TF Estimator. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.contrib import training as contrib_training

from REDACTED.tensorflow.contrib.tpu.python.tpu import bfloat16
from REDACTED.tensorflow.contrib.tpu.python.tpu import tpu_optimizer

from REDACTED.tensorflow_models.mlperf.models.rough.mlp_log import mlp_log
from REDACTED.tensorflow_models.mlperf.models.rough.ssd import dataloader
from REDACTED.tensorflow_models.mlperf.models.rough.ssd import ssd_architecture
from REDACTED.tensorflow_models.mlperf.models.rough.ssd import ssd_constants


def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: a Tensor with shape [batch_size, N, num_classes], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  batch_size, num_anchors, num_class = scores_in.get_shape().as_list()
  scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
  scores_trans = tf.reshape(scores_trans, [-1, num_anchors])

  top_k_scores, top_k_indices = tf.nn.top_k(
      scores_trans, k=pre_nms_num_detections, sorted=True)

  top_k_scores = tf.reshape(top_k_scores,
                            [batch_size, num_class, pre_nms_num_detections])
  top_k_indices = tf.reshape(top_k_indices,
                             [batch_size, num_class, pre_nms_num_detections])

  return tf.transpose(top_k_scores, [0, 2, 1]), tf.transpose(
      top_k_indices, [0, 2, 1])


def _filter_scores(scores, boxes, min_score=ssd_constants.MIN_SCORE):
  mask = scores > min_score
  scores = tf.where(mask, scores, tf.zeros_like(scores))
  boxes = tf.where(
      tf.tile(tf.expand_dims(mask, 2), (1, 1, 4)), boxes, tf.zeros_like(boxes))
  return scores, boxes


def non_max_suppression(scores_in,
                        boxes_in,
                        top_k_indices,
                        labels,
                        num_detections=ssd_constants.MAX_NUM_EVAL_BOXES):
  """Implement Non-maximum suppression.

  Args:
    scores_in: a Tensor with shape [batch_size,
      ssd_constants.MAX_NUM_EVAL_BOXES, num_classes]. The top
      ssd_constants.MAX_NUM_EVAL_BOXES box scores for each class.
    boxes_in: a Tensor with shape [batch_size, N, 4], which stacks box
      regression outputs on all feature levels. The N is the number of total
      anchors on all levels.
    top_k_indices: a Tensor with shape [batch_size,
      ssd_constants.MAX_NUM_EVAL_BOXES, num_classes]. The indices for these top
      boxes for each class.
    labels: labels tensor.
    num_detections: maximum output length.

  Returns:
    A tensor size of [batch_size, num_detections, 6] represents boxes, labels
    and scores after NMS.
  """

  _, _, num_classes = scores_in.get_shape().as_list()
  source_id = tf.cast(
      tf.tile(
          tf.expand_dims(labels[ssd_constants.SOURCE_ID], 1),
          [1, num_detections]), scores_in.dtype)
  raw_shape = tf.cast(
      tf.tile(
          tf.expand_dims(labels[ssd_constants.RAW_SHAPE], 1),
          [1, num_detections, 1]), scores_in.dtype)

  list_of_all_boxes = []
  list_of_all_scores = []
  list_of_all_classes = []
  # Skip background class.
  for class_i in range(1, num_classes, 1):
    boxes = tf.batch_gather(boxes_in, top_k_indices[:, :, class_i])
    class_i_scores = scores_in[:, :, class_i]
    class_i_scores, boxes = _filter_scores(class_i_scores, boxes)
    (class_i_post_scores,
     class_i_post_boxes) = ssd_architecture.non_max_suppression_padded(
         scores=tf.cast(class_i_scores, scores_in.dtype),
         boxes=tf.cast(boxes, scores_in.dtype),
         max_output_size=num_detections,
         iou_threshold=ssd_constants.OVERLAP_CRITERIA)
    class_i_classes = tf.fill(
        tf.shape(class_i_post_scores), ssd_constants.CLASS_INV_MAP[class_i])
    list_of_all_boxes.append(class_i_post_boxes)
    list_of_all_scores.append(class_i_post_scores)
    list_of_all_classes.append(class_i_classes)

  post_nms_boxes = tf.concat(list_of_all_boxes, axis=1)
  post_nms_scores = tf.concat(list_of_all_scores, axis=1)
  post_nms_classes = tf.concat(list_of_all_classes, axis=1)

  # sort all results.
  post_nms_scores, sorted_indices = tf.nn.top_k(
      tf.cast(post_nms_scores, scores_in.dtype), k=num_detections, sorted=True)

  post_nms_boxes = tf.gather(post_nms_boxes, sorted_indices, batch_dims=1)
  post_nms_classes = tf.gather(post_nms_classes, sorted_indices, batch_dims=1)
  detections_result = tf.stack([
      source_id,
      post_nms_boxes[:, :, 1] * raw_shape[:, :, 1],
      post_nms_boxes[:, :, 0] * raw_shape[:, :, 0],
      (post_nms_boxes[:, :, 3] - post_nms_boxes[:, :, 1]) * raw_shape[:, :, 1],
      (post_nms_boxes[:, :, 2] - post_nms_boxes[:, :, 0]) * raw_shape[:, :, 0],
      post_nms_scores,
      tf.cast(post_nms_classes, scores_in.dtype),
  ],
                               axis=2)

  return detections_result


def concat_outputs(cls_outputs, box_outputs):
  """Concatenate predictions into a single tensor.

  This function takes the dicts of class and box prediction tensors and
  concatenates them into a single tensor for comparison with the ground truth
  boxes and class labels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width,
      num_anchors * num_classses].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
  Returns:
    concatenanted cls_outputs and box_outputs.
  """
  assert set(cls_outputs.keys()) == set(box_outputs.keys())

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())
  batch_size = int(cls_outputs[keys[0]].shape[0])

  flat_cls = []
  flat_box = []
  for i, k in enumerate(keys):
    scale = ssd_constants.FEATURE_SIZES[i]
    split_shape = (ssd_constants.NUM_DEFAULTS[i], ssd_constants.NUM_CLASSES)
    assert cls_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape
    final_shape = (batch_size, scale**2 * split_shape[0], split_shape[1])
    flat_cls.append(
        tf.reshape(
            tf.transpose(
                tf.reshape(cls_outputs[k], intermediate_shape),
                (0, 3, 1, 2, 4)), final_shape))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4)
    assert box_outputs[k].shape[3] == split_shape[0] * split_shape[1]
    intermediate_shape = (batch_size, scale, scale) + split_shape
    final_shape = (batch_size, scale**2 * split_shape[0], split_shape[1])
    flat_box.append(
        tf.reshape(
            tf.transpose(
                tf.reshape(box_outputs[k], intermediate_shape),
                (0, 3, 1, 2, 4)), final_shape))

  return tf.concat(flat_cls, axis=1), tf.concat(flat_box, axis=1)


def _localization_loss(pred_locs, gt_locs, gt_labels, num_matched_boxes):
  """Computes the localization loss.

  Computes the localization loss using smooth l1 loss.
  Args:
    pred_locs: a dict from index to tensor of predicted locations. The shape
      of each tensor is [batch_size, num_anchors, 4].
    gt_locs: a list of tensors representing box regression targets in
      [batch_size, num_anchors, 4].
    gt_labels: a list of tensors that represents the classification groundtruth
      targets. The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets, used as the loss normalizater. The shape is [batch_size].
  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  keys = sorted(pred_locs.keys())
  box_loss = 0
  for i, k in enumerate(keys):
    gt_label = gt_labels[i]
    gt_loc = gt_locs[i]
    pred_loc = tf.reshape(pred_locs[k], gt_loc.shape)
    mask = tf.greater(gt_label, 0)
    float_mask = tf.cast(mask, tf.float32)

    smooth_l1 = tf.reduce_sum(
        tf.compat.v1.losses.huber_loss(
            gt_loc, pred_loc, reduction=tf.compat.v1.losses.Reduction.NONE),
        axis=-1)
    smooth_l1 = tf.multiply(smooth_l1, float_mask)
    box_loss = box_loss + tf.reduce_sum(
        smooth_l1, axis=list(range(1, smooth_l1.shape.ndims)))

  # TODO(taylorrobie): Confirm that normalizing by the number of boxes matches
  # reference
  return tf.reduce_mean(box_loss / num_matched_boxes)


@tf.custom_gradient
def _softmax_cross_entropy(logits, label):
  """Helper function to compute softmax cross entropy loss."""
  shifted_logits = logits - tf.expand_dims(tf.reduce_max(logits, -1), -1)
  exp_shifted_logits = tf.math.exp(shifted_logits)
  sum_exp = tf.reduce_sum(exp_shifted_logits, -1)
  log_sum_exp = tf.math.log(sum_exp)
  one_hot_label = tf.one_hot(label, ssd_constants.NUM_CLASSES)
  shifted_logits = tf.reduce_sum(shifted_logits * one_hot_label, -1)
  loss = log_sum_exp - shifted_logits

  def grad(dy):
    return (exp_shifted_logits / tf.expand_dims(sum_exp, -1) -
            one_hot_label) * tf.expand_dims(dy, -1), dy

  return loss, grad


def _topk_mask(scores, k):
  """Efficient implementation of topk_mask for TPUs."""

  def larger_count(data, limit):
    """Number of elements larger than limit along the most minor dimension."""
    ret = []
    for d in data:
      ret.append(
          tf.reduce_sum(
              tf.cast(d > tf.reshape(limit, [-1] + [1] * (d.shape.ndims - 1)),
                      tf.int32), axis=range(1, d.shape.ndims)))
    return tf.add_n(ret)

  def body(bit_index, value):
    """Body for the while loop executing the binary search."""
    new_value = tf.bitwise.bitwise_or(value,
                                      tf.bitwise.left_shift(1, bit_index))
    larger = larger_count(scores, tf.bitcast(new_value, tf.float32))
    next_value = tf.where(
        tf.logical_xor(larger >= k, kth_negative), new_value, value)
    return bit_index - 1, next_value

  kth_negative = (larger_count(scores, 0.0) < k)
  limit_sign = tf.where(kth_negative, tf.broadcast_to(1, kth_negative.shape),
                        tf.broadcast_to(0, kth_negative.shape))
  next_value = tf.bitwise.left_shift(limit_sign, 31)
  _, limit = tf.while_loop(lambda bit_index, _: bit_index >= 0, body,
                           (30, next_value))
  ret = []
  for score in scores:
    # Filter scores that are smaller than the threshold.
    ret.append(
        tf.where(
            score >= tf.reshape(
                tf.bitcast(limit, tf.float32), [-1] + [1] *
                (score.shape.ndims - 1)), tf.ones(score.shape),
            tf.zeros(score.shape)))
  return ret


def _classification_loss(pred_labels, gt_labels, num_matched_boxes):
  """Computes the classification loss.

  Computes the classification loss with hard negative mining.
  Args:
    pred_labels: a dict from index to tensor of predicted class. The shape
      of the tensor is [batch_size, num_anchors, num_classes].
    gt_labels: a list of tensor that represents the classification groundtruth
      targets. The shape is [batch_size, num_anchors, 1].
    num_matched_boxes: the number of anchors that are matched to a groundtruth
      targets. This is used as the loss normalizater.
  Returns:
    box_loss: a float32 representing total box regression loss.
  """
  keys = sorted(pred_labels.keys())
  cross_entropy = []
  for i, k in enumerate(keys):
    gt_label = gt_labels[i]
    pred_label = tf.reshape(
        pred_labels[k],
        gt_label.get_shape().as_list() + [ssd_constants.NUM_CLASSES])
    cross_entropy.append(_softmax_cross_entropy(pred_label, gt_label))

  float_mask = [tf.cast(gt_label > 0, tf.float32) for gt_label in gt_labels]

  # Hard example mining
  neg_masked_cross_entropy = [
      ce * (1 - m) for ce, m in zip(cross_entropy, float_mask)
  ]

  num_neg_boxes = tf.minimum(
      tf.cast(num_matched_boxes, tf.int32) * ssd_constants.NEGS_PER_POSITIVE,
      ssd_constants.NUM_SSD_BOXES)
  top_k_neg_mask = _topk_mask(neg_masked_cross_entropy, num_neg_boxes)

  class_loss = tf.add_n([
      tf.reduce_sum(tf.multiply(ce, fm + tm), axis=range(1, ce.shape.ndims))
      for ce, fm, tm in zip(cross_entropy, float_mask, top_k_neg_mask)
  ])

  return tf.reduce_mean(class_loss / num_matched_boxes)


def detection_loss(cls_outputs, box_outputs, labels):
  """Computes total detection loss.

  Computes total detection loss including box and class loss from all levels.
  Args:
    cls_outputs: an OrderDict with keys representing levels and values
      representing logits in [batch_size, height, width, num_anchors].
    box_outputs: an OrderDict with keys representing levels and values
      representing box regression targets in
      [batch_size, height, width, num_anchors * 4].
    labels: the dictionary that returned from dataloader that includes
      groundturth targets.
  Returns:
    total_loss: a float32 representing total loss reducing from class and box
      losses from all levels.
  """
  if isinstance(labels[ssd_constants.BOXES], dict):
    gt_boxes = list(labels[ssd_constants.BOXES].values())
    gt_classes = list(labels[ssd_constants.CLASSES].values())
  else:
    gt_boxes = [labels[ssd_constants.BOXES]]
    gt_classes = [labels[ssd_constants.CLASSES]]
    cls_outputs, box_outputs = concat_outputs(cls_outputs, box_outputs)
    cls_outputs = {'flatten': cls_outputs}
    box_outputs = {'flatten': box_outputs}

  box_loss = _localization_loss(
      box_outputs, gt_boxes, gt_classes,
      tf.reshape(labels[ssd_constants.NUM_MATCHED_BOXES], [-1]))
  class_loss = _classification_loss(
      cls_outputs, gt_classes,
      tf.reshape(labels[ssd_constants.NUM_MATCHED_BOXES], [-1]))

  return class_loss + box_loss


def update_learning_rate_schedule_parameters(params):
  """Updates params that are related to the learning rate schedule.

  Args:
    params: a parameter dictionary that includes learning_rate, lr_warmup_epoch,
      first_lr_drop_epoch, and second_lr_drop_epoch.
  """
  batch_size = params['batch_size'] * params['num_shards']
  # Learning rate is proportional to the batch size
  steps_per_epoch = params['num_examples_per_epoch'] / batch_size
  params['lr_warmup_step'] = int(params['lr_warmup_epoch'] * steps_per_epoch)
  params['first_lr_drop_step'] = int(
      params['first_lr_drop_epoch'] * steps_per_epoch)
  params['second_lr_drop_step'] = int(
      params['second_lr_drop_epoch'] * steps_per_epoch)


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
  batch_size = params['batch_size'] * params['num_shards']
  scaling_factor = batch_size / ssd_constants.DEFAULT_BATCH_SIZE
  mlp_log.mlperf_print('opt_learning_rate_warmup_factor', scaling_factor)
  mlp_log.mlperf_print('opt_learning_rate_warmup_steps', lr_warmup_step)
  adjusted_learning_rate = base_learning_rate * scaling_factor
  learning_rate = (tf.cast(global_step, dtype=tf.float32) /
                   lr_warmup_step) * adjusted_learning_rate
  lr_schedule = [[1.0, lr_warmup_step], [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             adjusted_learning_rate * mult)
  return learning_rate


class WeightDecayOptimizer(tf.train.Optimizer):
  """Wrapper to apply weight decay on gradients before all reduce."""

  def __init__(self, opt, weight_decay=0.0, name='WeightDecayOptimizer'):
    super(WeightDecayOptimizer, self).__init__(False, name)
    self._opt = opt
    self._weight_decay = weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None:
        summed_grads_and_vars.append((grad, var))
      else:
        summed_grads_and_vars.append((grad + var * self._weight_decay, var))
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)


def ssd_model_fn(params, features, labels, is_training):
  """Model defination for the SSD model based on ResNet-50.

  Args:
    params: the dictionary defines hyperparameters of model. The default
      settings are in default_hparams function in this file.
    features: the input image tensor with shape [batch_size, height, width, 3].
      The height and width are fixed and equal.
    labels: the input labels in a dictionary. The labels include class targets
      and box targets which are dense label maps. The labels are generated from
      get_input_fn function in data/dataloader.py
    is_training: whether this is for training.

  Returns:
    [train_op, predictions]
  """

  # Manually apply the double transpose trick for training data.
  if params['transpose_input'] and is_training:
    if params['batch_size'] > 8:
      features = tf.transpose(features, [3, 0, 1, 2])
    else:
      features = tf.transpose(features, [2, 0, 1, 3])
    labels[ssd_constants.BOXES] = tf.transpose(labels[ssd_constants.BOXES],
                                               [2, 0, 1])

  if params['conv0_space_to_depth']:
    block_size = ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE
    # Depth increased by block_size * block_size from width and height
    # in space to depth transform in conv0.
    scale_size = block_size**2
    features -= tf.constant(
        ssd_constants.NORMALIZATION_MEAN,
        shape=[1, 1, 3 * scale_size],
        dtype=features.dtype)

    features /= tf.constant(
        ssd_constants.NORMALIZATION_STD,
        shape=[1, 1, 3 * scale_size],
        dtype=features.dtype)
  else:
    features -= tf.constant(
        ssd_constants.NORMALIZATION_MEAN, shape=[1, 1, 3], dtype=features.dtype)

    features /= tf.constant(
        ssd_constants.NORMALIZATION_STD, shape=[1, 1, 3], dtype=features.dtype)

  if params['use_bfloat16']:
    with bfloat16.bfloat16_scope():
      cls_outputs, box_outputs = ssd_architecture.ssd(features, params,
                                                      is_training)
      levels = cls_outputs.keys()
      for level in levels:
        cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
        box_outputs[level] = tf.cast(box_outputs[level], tf.float32)
  else:
    cls_outputs, box_outputs = ssd_architecture.ssd(features, params,
                                                    is_training)
    levels = cls_outputs.keys()

  # First check if it is in PREDICT mode.
  if not is_training:
    flattened_cls, flattened_box = concat_outputs(cls_outputs, box_outputs)

    anchors = tf.convert_to_tensor(dataloader.DefaultBoxes()('xywh'))

    flattened_box /= tf.reshape(
        tf.convert_to_tensor(ssd_constants.BOX_CODER_SCALES), [1, 1, 4])
    ycenter_a, xcenter_a, ha, wa = tf.unstack(anchors, axis=-1)
    ty, tx, th, tw = tf.unstack(flattened_box, axis=-1)
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    decoded_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    pred_scores = tf.nn.softmax(flattened_cls, axis=2)

    pred_scores, indices = select_top_k_scores(pred_scores,
                                               ssd_constants.MAX_NUM_EVAL_BOXES)
    detections = non_max_suppression(
        scores_in=pred_scores,
        boxes_in=decoded_boxes,
        top_k_indices=indices,
        labels=labels)
    predictions = dict(detections=detections)

    if ssd_constants.IS_PADDED in labels:
      predictions[ssd_constants.IS_PADDED] = labels[ssd_constants.IS_PADDED]

    return None, predictions

  # Set up training loss and learning rate.
  update_learning_rate_schedule_parameters(params)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = learning_rate_schedule(params, global_step)
  # cls_loss and box_loss are for logging. only total_loss is optimized.
  total_loss = detection_loss(cls_outputs, box_outputs, labels)

  optimizer = tf.train.MomentumOptimizer(
      learning_rate, momentum=ssd_constants.MOMENTUM)
  optimizer = tpu_optimizer.CrossShardOptimizer(
      WeightDecayOptimizer(optimizer, params['weight_decay']))

  # Batch norm requires update_ops to be added as a train_op dependency.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    return optimizer.minimize(total_loss, global_step), None


def default_hparams():
  return contrib_training.HParams(
      use_bfloat16=True,
      num_examples_per_epoch=120000,
      lr_warmup_epoch=1.0,
      first_lr_drop_epoch=40,
      second_lr_drop_epoch=50,
      weight_decay=ssd_constants.WEIGHT_DECAY,
      base_learning_rate=ssd_constants.BASE_LEARNING_RATE,
      distributed_group_size=1,
      tpu_slice_row=-1,
      tpu_slice_col=-1,
      dbn_tile_row=-1,  # number of rows in each distributed batch norm group.
      dbn_tile_col=-1,  # number of cols in each distributed batch norm group.
      eval_every_checkpoint=False,
      transpose_input=True,
      conv0_space_to_depth=False,
      eval_samples=ssd_constants.EVAL_SAMPLES,
      use_spatial_partitioning=False,
  )
