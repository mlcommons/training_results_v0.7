"""Model defination for the SSD Model.

Defines model_fn of SSD with JAX. The model_fn includes SSD
model architecture, loss function, learning rate schedule, and evaluation
procedure.

SSD: Single Shot MultiBox Detector
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
https://arxiv.org/abs/1512.02325
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools as it
import math

from flax import nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp

from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import nms
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_constants


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  """Returns a one-hot tensor.

  Args:
    labels: Ground truth label tensor.
    num_classes: int value for the number classes.
    on_value: A scalar defining the value to fill in output when `indices[j]
      = i`. (default: 1)
    off_value: A scalar defining the value to fill in output when `indices[j]
      != i`. (default: 0)
  Returns:
    output: The one-hot tensor.
  """
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


def huber_loss(labels, predictions, weights=1.0, delta=1.0):
  """Adds a Huber Loss term to the training procedure.

  For each value x in `error=labels-predictions`, the following is calculated:
  ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`.
  See: https://en.wikipedia.org/wiki/Huber_loss
  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  `[batch_size]`, then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.
  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    delta: `float`, the point where the huber loss function
      changes from a quadratic to linear.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or
     `predictions` is None.
  """
  if labels is None:
    raise ValueError('labels must not be None.')
  if predictions is None:
    raise ValueError('predictions must not be None.')
  predictions = predictions.astype(jnp.float32)
  labels = labels.astype(jnp.float32)
  # TODO(deveci): check for shape compabitibilty
  # predictions.get_shape().assert_is_compatible_with(labels.get_shape())
  error = jnp.subtract(predictions, labels)
  abs_error = jnp.abs(error)
  quadratic = jnp.minimum(abs_error, delta)
  # The following expression is the same in value as
  # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
  # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
  # This is necessary to avoid doubling the gradient, since there is already a
  # nonzero contribution to the gradient from the quadratic term.
  linear = jnp.subtract(abs_error, quadratic)
  losses = jnp.add(
      jnp.multiply(
          0.5,
          jnp.multiply(quadratic, quadratic)),
      jnp.multiply(delta, linear))

  input_dtype = losses.dtype
  losses = losses.astype(jnp.float32)
  weights = jnp.asarray(weights, jnp.float32)
  weighted_losses = jnp.multiply(losses, weights)
  loss = weighted_losses.astype(input_dtype)
  return loss


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
    pred_loc = jnp.reshape(pred_locs[k], gt_loc.shape)
    mask = jnp.greater(gt_label, 0)
    float_mask = mask.astype(jnp.float32)

    smooth_l1 = jnp.sum(huber_loss(gt_loc, pred_loc), axis=-1)
    smooth_l1 = jnp.multiply(smooth_l1, float_mask)
    box_loss = box_loss + jnp.sum(
        smooth_l1, axis=list(range(1, len(smooth_l1.shape))))
  return jnp.mean(box_loss / num_matched_boxes)


def softmax_cross_entropy(logits, onehot_labels):
  """Returns a cross-entropy loss tensor.

  Note that `onehot_labels` and `logits` must have the same shape,
  e.g. `[batch_size, num_classes]`. This does not perform a reduction on loss,
  e.g., loss is a `Tensor` of shape `[batch_size]`.

  Args:
    logits: Float-like logits outputs of the network.
    onehot_labels: Float-like one-hot-encoded labels.
  Returns:
    Loss `Tensor` of the same type as `logits`, which has shape `[batch_size]`.
  """
  log_softmax = -nn.log_softmax(logits)
  return jnp.sum(log_softmax * onehot_labels, axis=-1)


@jax.custom_gradient
def _softmax_cross_entropy(logits, label):
  """Helper function to compute softmax cross entropy loss."""
  shifted_logits = logits - jnp.expand_dims(jnp.max(logits, -1), -1)
  exp_shifted_logits = jnp.exp(shifted_logits)
  sum_exp = jnp.sum(exp_shifted_logits, -1)
  log_sum_exp = jnp.log(sum_exp)
  one_hot_label = onehot(label, ssd_constants.NUM_CLASSES)
  shifted_logits = jnp.sum(shifted_logits * one_hot_label, -1)
  loss = log_sum_exp - shifted_logits

  def grad(dy):
    return (exp_shifted_logits / jnp.expand_dims(sum_exp, -1) -
            one_hot_label) * jnp.expand_dims(dy, -1), dy
  return loss, grad


def _topk_mask(scores, k):
  """Efficient implementation of topk_mask for TPUs."""
  def bitcast(data, newtype):
    return jax.lax.bitcast_convert_type(data, newtype)

  def larger_count(data, limit):
    """Number of elements larger than limit along the most minor dimension."""
    ret = []
    for d in data:
      ret.append(
          jnp.sum(
              (d > jnp.reshape(limit, [-1] + [1] * (len(d.shape) - 1))).astype(
                  jnp.int32), axis=list(range(1, len(d.shape)))))
    return sum(ret)

  def body(args, _):
    """Body for the while loop executing the binary search."""
    bit_index, value = args
    new_value = jnp.bitwise_or(value, jnp.left_shift(1, bit_index))
    larger = larger_count(scores, bitcast(new_value, jnp.float32))
    next_value = jnp.where(
        jnp.logical_xor(larger >= k, kth_negative), new_value, value)
    return (bit_index - 1, next_value), None

  kth_negative = (larger_count(scores, jnp.array(0.0)) < k)
  limit_sign = jnp.where(kth_negative, jnp.broadcast_to(1, kth_negative.shape),
                         jnp.broadcast_to(0, kth_negative.shape))
  next_value = jnp.left_shift(limit_sign, 31)
  bit_index = jnp.array(30)
  (_, limit), _ = lax.scan(body, (bit_index, next_value), None, length=31)
  ret = []
  for score in scores:
    # Filter scores that are smaller than the threshold.
    ret.append(
        jnp.where(
            score >= jnp.reshape(
                bitcast(limit, jnp.float32), [-1] + [1] *
                (len(score.shape) - 1)), jnp.ones(score.shape),
            jnp.zeros(score.shape)))

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
    pred_label = jnp.reshape(
        pred_labels[k], list(gt_label.shape) + [ssd_constants.NUM_CLASSES])
    cross_entropy.append(_softmax_cross_entropy(pred_label, gt_label))

  float_mask = [(gt_label > 0).astype(jnp.float32) for gt_label in gt_labels]

  # Hard example mining
  neg_masked_cross_entropy = [
      ce * (1 - m) for ce, m in zip(cross_entropy, float_mask)
  ]

  num_neg_boxes = jnp.minimum(
      num_matched_boxes.astype(jnp.int32) * ssd_constants.NEGS_PER_POSITIVE,
      ssd_constants.NUM_SSD_BOXES)
  top_k_neg_mask = _topk_mask(neg_masked_cross_entropy, num_neg_boxes)

  class_loss = sum([
      jnp.sum(jnp.multiply(ce, fm + tm), axis=list(range(1, len(ce.shape))))
      for ce, fm, tm in zip(cross_entropy, float_mask, top_k_neg_mask)
  ])

  return jnp.mean(class_loss / num_matched_boxes)


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
    concatenanted cls_outputs (batch_size, num_anchors, num_classes) and
      box_outputs (batch_size, num_anchors, 4).
  """
  if set(cls_outputs.keys()) != set(box_outputs.keys()):
    raise ValueError('Class output and box output keys do not match.')

  # This sort matters. The labels assume a certain order based on
  # ssd_constants.FEATURE_SIZES, and this sort matches that convention.
  keys = sorted(cls_outputs.keys())
  batch_size = int(cls_outputs[keys[0]].shape[0])

  flat_cls = []
  flat_box = []

  for i, k in enumerate(keys):
    scale = ssd_constants.FEATURE_SIZES[i]
    split_shape = (ssd_constants.NUM_DEFAULTS[i], ssd_constants.NUM_CLASSES)
    if cls_outputs[k].shape[3] != split_shape[0] * split_shape[1]:
      raise ValueError(
          'Unexpected class shapes at level:%d, cls_outputs:%d, expected:%d' %
          (k, cls_outputs[k].shape[3], split_shape[0] * split_shape[1]))
    intermediate_shape = (batch_size, scale, scale) + split_shape
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1])

    flat_cls.append(
        jnp.reshape(
            jnp.transpose(
                jnp.reshape(cls_outputs[k], intermediate_shape),
                (0, 3, 1, 2, 4)),
            final_shape))

    split_shape = (ssd_constants.NUM_DEFAULTS[i], 4)
    if box_outputs[k].shape[3] != split_shape[0] * split_shape[1]:
      raise ValueError(
          'Unexpected box shapes at level:%d, box_outputs:%d, expected:%d' %
          (k, box_outputs[k].shape[3], split_shape[0] * split_shape[1]))
    intermediate_shape = (batch_size, scale, scale) + split_shape
    final_shape = (batch_size, scale ** 2 * split_shape[0], split_shape[1])
    flat_box.append(
        jnp.reshape(
            jnp.transpose(
                jnp.reshape(box_outputs[k], intermediate_shape),
                (0, 3, 1, 2, 4)),
            final_shape))

  return jnp.concatenate(flat_cls, axis=1), jnp.concatenate(flat_box, axis=1)


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
    cls_loss: a float32 representing total class loss.
    box_loss: a float32 representing total box regression loss.
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

  box_loss = _localization_loss(box_outputs, gt_boxes, gt_classes,
                                labels[ssd_constants.NUM_MATCHED_BOXES])

  class_loss = _classification_loss(cls_outputs, gt_classes,
                                    labels[ssd_constants.NUM_MATCHED_BOXES])

  return class_loss + box_loss, class_loss, box_loss


def box_decode(flattened_box):
  """Decode boxes.

  Args:
    flattened_box: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  anchors = jnp.array(DefaultBoxes()('xywh'))

  flattened_box /= jnp.reshape(jnp.array(ssd_constants.BOX_CODER_SCALES),
                               [1, 1, 4])

  ycenter_a, xcenter_a, ha, wa = jnp.split(anchors, anchors.shape[-1], axis=-1)
  ycenter_a = jnp.squeeze(ycenter_a, axis=-1)
  xcenter_a = jnp.squeeze(xcenter_a, axis=-1)
  ha = jnp.squeeze(ha, axis=-1)
  wa = jnp.squeeze(wa, axis=-1)

  ty, tx, th, tw = jnp.split(flattened_box, flattened_box.shape[-1], axis=-1)
  ty = jnp.squeeze(ty, axis=-1)
  tx = jnp.squeeze(tx, axis=-1)
  th = jnp.squeeze(th, axis=-1)
  tw = jnp.squeeze(tw, axis=-1)

  w = jnp.exp(tw) * wa
  h = jnp.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  decoded_boxes = jnp.stack([ymin, xmin, ymax, xmax], axis=-1)
  return decoded_boxes


class DefaultBoxes(object):
  """Default bounding boxes for 300x300 5 layer SSD.

  Default bounding boxes generation follows the order of (W, H, anchor_sizes).
  Therefore, the tensor converted from DefaultBoxes has a shape of
  [anchor_sizes, H, W, 4]. The last dimension is the box coordinates; 'ltrb'
  is [ymin, xmin, ymax, xmax] while 'xywh' is [cy, cx, h, w].
  """

  def __init__(self):
    fk = ssd_constants.IMAGE_SIZE / onp.array(ssd_constants.STEPS)

    self.default_boxes = []
    # size of feature and number of feature
    for idx, feature_size in enumerate(ssd_constants.FEATURE_SIZES):
      sk1 = ssd_constants.SCALES[idx] / ssd_constants.IMAGE_SIZE
      sk2 = ssd_constants.SCALES[idx+1] / ssd_constants.IMAGE_SIZE
      sk3 = math.sqrt(sk1*sk2)
      all_sizes = [(sk1, sk1), (sk3, sk3)]

      for alpha in ssd_constants.ASPECT_RATIOS[idx]:
        w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
        all_sizes.append((w, h))
        all_sizes.append((h, w))

      assert len(all_sizes) == ssd_constants.NUM_DEFAULTS[idx]

      for w, h in all_sizes:
        for i, j in it.product(range(feature_size), repeat=2):
          cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
          box = tuple(onp.clip(k, 0, 1) for k in (cy, cx, h, w))
          self.default_boxes.append(box)

    assert len(self.default_boxes) == ssd_constants.NUM_SSD_BOXES

    def to_ltrb(cy, cx, h, w):
      return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

    # For IoU calculation
    self.default_boxes_ltrb = tuple(to_ltrb(*i) for i in self.default_boxes)

  def __call__(self, order='ltrb'):
    if order == 'ltrb': return self.default_boxes_ltrb
    if order == 'xywh': return self.default_boxes


def select_top_k_scores(scores_in, pre_nms_num_detections=5000):
  """Select top_k scores and indices for each class.

  Args:
    scores_in: A tensor with shape [batch_size, N, num_classes], which stacks
      class logit outputs on all feature levels. The N is the number of total
      anchors on all levels. The num_classes is the number of classes predicted
      by the model.
    pre_nms_num_detections: Number of candidates before NMS.

  Returns:
    scores and indices: Tensors with shape [batch_size, pre_nms_num_detections,
      num_classes].
  """
  batch_size, num_anchors, num_class = list(scores_in.shape)
  scores_trans = jnp.transpose(scores_in, axes=[0, 2, 1])
  scores_trans = jnp.reshape(scores_trans, [-1, num_anchors])

  top_k_scores, top_k_indices = lax.top_k(scores_trans,
                                          k=pre_nms_num_detections)
  top_k_scores = jnp.reshape(top_k_scores,
                             [batch_size, num_class, pre_nms_num_detections])
  top_k_indices = jnp.reshape(top_k_indices,
                              [batch_size, num_class, pre_nms_num_detections])

  return jnp.transpose(top_k_scores, [0, 2, 1]), jnp.transpose(
      top_k_indices, [0, 2, 1])


def _filter_scores(scores, boxes, min_score=ssd_constants.MIN_SCORE):
  mask = scores > min_score
  scores = jnp.where(mask, scores, jnp.zeros_like(scores))
  tiled = jnp.tile(jnp.expand_dims(mask, 2), (1, 1, 4))
  boxes = jnp.where(
      tiled,
      boxes, jnp.zeros_like(boxes))
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
    A tensor size of [batch_size, num_detections, 7] represents boxes, labels
    and scores after NMS.
  """

  _, _, num_classes = list(scores_in.shape)
  source_id = jnp.tile(jnp.expand_dims(labels[ssd_constants.SOURCE_ID], 1),
                       [1, num_detections]).astype(scores_in.dtype)
  raw_shape = jnp.tile(jnp.expand_dims(labels[ssd_constants.RAW_SHAPE], 1),
                       [1, num_detections, 1]).astype(scores_in.dtype)

  list_of_all_boxes = []
  list_of_all_scores = []
  list_of_all_classes = []
  # Skip background class.
  for class_i in range(1, num_classes, 1):
    # TODO(deveci): below may not be performant
    # suggestion(wangtao): convert to 2D and try numpy indexing.
    # boxes = jax.vmap(lambda x, i: x[i])(boxes_in,
    #                                     top_k_indices[:, :, class_i]))
    boxes = jnp.stack(
        [x[y] for x, y in zip(boxes_in, top_k_indices[:, :, class_i])])

    class_i_scores = scores_in[:, :, class_i]
    class_i_scores, boxes = _filter_scores(class_i_scores, boxes)
    (class_i_post_scores, class_i_post_boxes) = nms.non_max_suppression_padded(
        scores=class_i_scores.astype(scores_in.dtype),
        boxes=boxes.astype(scores_in.dtype),
        max_output_size=num_detections,
        iou_threshold=ssd_constants.OVERLAP_CRITERIA)
    class_i_classes = jnp.full(class_i_post_scores.shape,
                               ssd_constants.CLASS_INV_MAP[class_i])
    list_of_all_boxes.append(class_i_post_boxes)
    list_of_all_scores.append(class_i_post_scores)
    list_of_all_classes.append(class_i_classes)

  post_nms_boxes = jnp.concatenate(list_of_all_boxes, axis=1)
  post_nms_scores = jnp.concatenate(list_of_all_scores, axis=1)
  post_nms_classes = jnp.concatenate(list_of_all_classes, axis=1)

  # sort all results.
  post_nms_scores, sorted_indices = lax.top_k(
      post_nms_scores.astype(scores_in.dtype), k=num_detections)
  # TODO(deveci): below may not be performant
  # suggestion(wangtao): convert to 2D and try numpy indexing.
  # post_nms_boxes = tf.gather(post_nms_boxes, sorted_indices, batch_dims=1)
  post_nms_boxes = jnp.stack([post_nms_boxes[i][sorted_indices[i]]
                              for i in range(len(post_nms_boxes))])
  # TODO(deveci): below may not be performant
  # suggestion(wangtao): convert to 2D and try numpy indexing.
  # post_nms_classes = tf.gather(post_nms_classes, sorted_indices, batch_dims=1)
  post_nms_classes = jnp.stack([post_nms_classes[i][sorted_indices[i]]
                                for i in range(len(post_nms_classes))])
  detections_result = jnp.stack([
      source_id,
      post_nms_boxes[:, :, 1] * raw_shape[:, :, 1],
      post_nms_boxes[:, :, 0] * raw_shape[:, :, 0],
      (post_nms_boxes[:, :, 3] - post_nms_boxes[:, :, 1]) * raw_shape[:, :, 1],
      (post_nms_boxes[:, :, 2] - post_nms_boxes[:, :, 0]) * raw_shape[:, :, 0],
      post_nms_scores,
      post_nms_classes.astype(scores_in.dtype),],
                                axis=2)

  return detections_result
