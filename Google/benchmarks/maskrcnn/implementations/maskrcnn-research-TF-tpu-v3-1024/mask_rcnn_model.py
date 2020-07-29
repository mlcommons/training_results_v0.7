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
"""Model defination for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow.compat.v1 as tf
from REDACTED.tensorflow.contrib import tpu as contrib_tpu

from REDACTED.mask_rcnn import anchors
from REDACTED.mask_rcnn import fpn
from REDACTED.mask_rcnn import losses
from REDACTED.mask_rcnn import lr_policy
from REDACTED.mask_rcnn import mask_rcnn_architecture
from REDACTED.mask_rcnn import mask_rcnn_params
from REDACTED.mask_rcnn import post_processing
from REDACTED.mlp_log import mlp_log

_WEIGHT_DECAY = 1e-4


class WeightDecayOptimizer(tf.train.Optimizer):
  """Wrapper to apply weight decay on gradients before all reduce."""

  def __init__(self, opt, weight_decay=0.0, name='WeightDecayOptimizer'):
    super(WeightDecayOptimizer, self).__init__(False, name)
    self._opt = opt
    self._weight_decay = weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None or 'batch_normalization' in var.name or 'bias' in var.name:
        summed_grads_and_vars.append((grad, var))
      else:
        summed_grads_and_vars.append((grad + var * self._weight_decay, var))
    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)


class MaskRcnnModelFn(object):
  """Mask-Rcnn model function."""

  def __init__(self, params):
    self.params = params

  def remove_variables(self, variables, resnet_depth=50):
    """Removes low-level variables from the training.

    Removing low-level parameters (e.g., initial convolution layer) from
    training usually leads to higher training speed and slightly better testing
    accuracy. The intuition is that the low-level architecture
    (e.g., ResNet-50) is able to capture low-level features such as edges;
    therefore, it does not need to be fine-tuned for the detection task.

    Args:
      variables: all the variables in training
      resnet_depth: the depth of ResNet model

    Returns:
      A list containing variables for training.

    """
    # Freeze at conv2 based on reference model.
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/ResNet.py  # pylint: disable=line-too-long
    remove_list = []
    prefix = 'resnet{}/'.format(resnet_depth)
    remove_list.append(prefix + 'conv2d/')
    for i in range(1, 11):
      remove_list.append(prefix + 'conv2d_{}/'.format(i))

    # All batch normalization variables are frozen during training.
    def _is_kept(variable):
      return (all(rm_str not in variable.name for rm_str in remove_list) and
              'batch_normalization' not in variable.name)

    return list(filter(_is_kept, variables))

  def get_learning_rate(self, global_step):
    """Sets up learning rate schedule."""
    learning_rate = lr_policy.learning_rate_schedule(
        self.params['learning_rate'], self.params['lr_warmup_init'],
        self.params['lr_warmup_step'], self.params['first_lr_drop_step'],
        self.params['second_lr_drop_step'], global_step)
    mlp_log.mlperf_print(
        key='opt_base_learning_rate', value=self.params['learning_rate'])
    mlp_log.mlperf_print(
        key='opt_learning_rate_warmup_steps',
        value=self.params['lr_warmup_step'])
    mlp_log.mlperf_print(
        key='opt_learning_rate_warmup_factor',
        value=self.params['learning_rate'] / self.params['lr_warmup_step'])
    return learning_rate

  def get_optimizer(self, learning_rate):
    """Defines the optimizer."""
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=self.params['momentum'])
    optimizer = WeightDecayOptimizer(optimizer, _WEIGHT_DECAY)
    return contrib_tpu.CrossShardOptimizer(optimizer)

  def _model_outputs(self, features, labels, image_size, is_training):
    """Generates outputs from the model."""
    all_anchors = anchors.Anchors(self.params['min_level'],
                                  self.params['max_level'],
                                  self.params['num_scales'],
                                  self.params['aspect_ratios'],
                                  self.params['anchor_scale'], image_size)

    if self.params['conv0_space_to_depth_block_size'] != 0:
      image_size = tuple(x // self.params['conv0_space_to_depth_block_size']
                         for x in image_size)

    if self.params['transpose_input']:
      images = tf.reshape(features['images'], [
          image_size[0], image_size[1], -1,
          3 * self.params['conv0_space_to_depth_block_size'] *
          self.params['conv0_space_to_depth_block_size']
      ])
      images = tf.transpose(images, [2, 0, 1, 3])
    else:
      images = tf.reshape(features['images'], [
          -1, image_size[0], image_size[1],
          3 * self.params['conv0_space_to_depth_block_size'] *
          self.params['conv0_space_to_depth_block_size']
      ])

    fpn_feats = fpn.resnet_fpn(images, self.params['min_level'],
                               self.params['max_level'],
                               self.params['resnet_depth'],
                               self.params['conv0_kernel_size'],
                               self.params['conv0_space_to_depth_block_size'],
                               self.params['is_training_bn'])

    rpn_score_outputs, rpn_box_outputs = mask_rcnn_architecture.rpn_net(
        fpn_feats, self.params['min_level'], self.params['max_level'],
        len(self.params['aspect_ratios'] * self.params['num_scales']))

    if not is_training:
      # Use TEST.NMS in the reference for this value. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L227  # pylint: disable=line-too-long

      # The mask branch takes inputs from different places in training vs in
      # eval/predict. In training, the mask branch uses proposals combined
      # with labels to produce both mask outputs and targets. At test time,
      # it uses the post-processed predictions to generate masks.
      # Generate detections one image at a time.
      (class_outputs,
       box_outputs, box_rois) = mask_rcnn_architecture.faster_rcnn(
           fpn_feats, rpn_score_outputs, rpn_box_outputs, all_anchors,
           features['image_info'], is_training, self.params)
      batch_size, _, _ = class_outputs.get_shape().as_list()
      detections = []
      softmax_class_outputs = tf.nn.softmax(class_outputs)
      for i in range(batch_size):
        detections.append(
            post_processing.generate_detections_per_image_op(
                softmax_class_outputs[i], box_outputs[i], box_rois[i],
                features['source_ids'][i], features['image_info'][i],
                self.params['test_detections_per_image'],
                self.params['test_rpn_post_nms_topn'], self.params['test_nms'],
                self.params['bbox_reg_weights']))
      detections = tf.stack(detections, axis=0)
      mask_outputs = mask_rcnn_architecture.mask_rcnn(
          fpn_feats, is_training, self.params, detections=detections)
      return {'detections': detections, 'mask_outputs': mask_outputs}
    else:
      (class_outputs, box_outputs, box_rois, class_targets, box_targets,
       proposal_to_label_map) = mask_rcnn_architecture.faster_rcnn(
           fpn_feats, rpn_score_outputs, rpn_box_outputs, all_anchors,
           features['image_info'], is_training, self.params, labels)
      encoded_box_targets = mask_rcnn_architecture.encode_box_targets(
          box_rois, box_targets, class_targets, self.params['bbox_reg_weights'])
      (mask_outputs, select_class_targets,
       mask_targets) = mask_rcnn_architecture.mask_rcnn(fpn_feats, is_training,
                                                        self.params, labels,
                                                        class_targets,
                                                        box_targets, box_rois,
                                                        proposal_to_label_map)
      return {
          'rpn_score_outputs': rpn_score_outputs,
          'rpn_box_outputs': rpn_box_outputs,
          'class_outputs': class_outputs,
          'box_outputs': box_outputs,
          'class_targets': class_targets,
          'box_targets': encoded_box_targets,
          'box_rois': box_rois,
          'select_class_targets': select_class_targets,
          'mask_outputs': mask_outputs,
          'mask_targets': mask_targets,}

  def get_model_outputs(self, features, labels, image_size, is_training):
    """A wrapper to generate outputs from the model.

    Args:
      features: the input image tensor and auxiliary information, such as
        `image_info` and `source_ids`. The image tensor has a shape of
        [batch_size, height, width, 3]. The height and width are fixed and
        equal.
      labels: the input labels in a dictionary. The labels include score targets
        and box targets which are dense label maps. See dataloader.py for more
        details.
      image_size: an integer tuple (height, width) representing the image shape.
      is_training: whether if this is training.

    Returns:
      The outputs from model (all casted to tf.float32).
    """

    if self.params['use_bfloat16']:
      with contrib_tpu.bfloat16_scope():
        outputs = self._model_outputs(features, labels, image_size, is_training)

        def _cast_outputs_to_float(d):
          for k, v in six.iteritems(d):
            if isinstance(v, dict):
              _cast_outputs_to_float(v)
            else:
              d[k] = tf.cast(v, tf.float32)
        _cast_outputs_to_float(outputs)
    else:
      outputs = self._model_outputs(features, labels, image_size, is_training)
    return outputs

  def predict(self, features, labels):
    """Generates predicitons."""
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      def branch_fn(image_size):
        return self.get_model_outputs(features, labels, image_size, False)

      model_outputs = tf.cond(
          tf.less(features['image_info'][0][3], features['image_info'][0][4]),
          lambda: branch_fn(self.params['image_size']),
          lambda: branch_fn(self.params['image_size'][::-1]))

      def scale_detections_to_original_image_size(detections, image_info):
        """Maps [y1, x1, y2, x2] -> [x1, y1, w, h] and scales detections."""
        batch_size, _, _ = detections.get_shape().as_list()
        image_ids, y_min, x_min, y_max, x_max, scores, classes = tf.split(
            value=detections, num_or_size_splits=7, axis=2)
        image_scale = tf.reshape(image_info[:, 2], [batch_size, 1, 1])
        scaled_height = (y_max - y_min) * image_scale
        scaled_width = (x_max - x_min) * image_scale
        scaled_y = y_min * image_scale
        scaled_x = x_min * image_scale
        detections = tf.concat(
            [image_ids, scaled_x, scaled_y, scaled_width, scaled_height, scores,
             classes],
            axis=2)
        return detections

      predictions = {}
      predictions['detections'] = scale_detections_to_original_image_size(
          model_outputs['detections'], features['image_info'])
      predictions['mask_outputs'] = tf.nn.sigmoid(model_outputs['mask_outputs'])
      predictions['image_info'] = features['image_info']
      if mask_rcnn_params.IS_PADDED in features:
        predictions[mask_rcnn_params.IS_PADDED] = features[
            mask_rcnn_params.IS_PADDED]
      tf.logging.info(features)

      return predictions

  def get_loss(self, model_outputs, labels):
    """Generates the loss function."""
    # score_loss and box_loss are for logging. only total_loss is optimized.
    total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
        model_outputs['rpn_score_outputs'], model_outputs['rpn_box_outputs'],
        labels, self.params)

    total_fast_rcnn_loss, class_loss, box_loss = losses.fast_rcnn_loss(
        model_outputs['class_outputs'], model_outputs['box_outputs'],
        model_outputs['class_targets'], model_outputs['box_targets'],
        self.params)

    mask_loss = losses.mask_rcnn_loss(model_outputs['mask_outputs'],
                                      model_outputs['mask_targets'],
                                      model_outputs['select_class_targets'],
                                      self.params)

    total_loss = (total_rpn_loss + total_fast_rcnn_loss + mask_loss)

    return [total_loss, mask_loss, total_fast_rcnn_loss, class_loss, box_loss,
            total_rpn_loss, rpn_score_loss, rpn_box_loss]

  def train_op(self, features, labels, image_size, optimizer):
    """Generates train op."""
    model_outputs = self.get_model_outputs(features, labels, image_size, True)

    var_list = self.remove_variables(tf.trainable_variables(),
                                     self.params['resnet_depth'])
    all_losses = self.get_loss(model_outputs, labels)

    gradients, _ = zip(*optimizer.compute_gradients(
        all_losses[0], var_list, colocate_gradients_with_ops=True))
    ret = []
    for grad, var in zip(gradients, var_list):
      if 'beta' in var.name or 'bias' in var.name:
        grad = 2.0 * grad
      ret.append(grad)
    return ret

  def train(self, features, labels):
    """A wrapper for tf.cond."""

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      global_step = tf.train.get_or_create_global_step()
      learning_rate = self.get_learning_rate(global_step)
      optimizer = self.get_optimizer(learning_rate)

      def branch_fn(image_size, optimizer):
        return self.train_op(features, labels, image_size, optimizer)

      grads = tf.cond(
          tf.less(features['image_info'][0][3], features['image_info'][0][4]),
          lambda: branch_fn(self.params['image_size'], optimizer),
          lambda: branch_fn(self.params['image_size'][::-1], optimizer))

      variables = self.remove_variables(tf.trainable_variables(),
                                        self.params['resnet_depth'])

      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        return optimizer.apply_gradients(
            zip(grads, variables), global_step=global_step)

  def __call__(self, features, labels, is_training):
    """Model defination for the Mask-RCNN model based on ResNet.

    Args:
      features: the input image tensor and auxiliary information, such as
        `image_info` and `source_ids`. The image tensor has a shape of
        [batch_size, height, width, 3]. The height and width are fixed and
        equal.
      labels: the input labels in a dictionary. The labels include score targets
        and box targets which are dense label maps. The labels are generated
        from get_input_fn function in data/dataloader.py
      is_training: whether if this is training.

    Returns:
      [train_op, predictions]
    """
    if not is_training:
      return None, self.predict(features, labels)
    else:
      return self.train(features, labels), None
