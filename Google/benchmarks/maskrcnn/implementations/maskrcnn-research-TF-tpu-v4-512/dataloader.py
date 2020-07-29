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
"""Data loader and processing.

Defines input_fn of Mask-RCNN for TF Estimator. The input_fn includes training
data for category classification, bounding box regression, and number of
positive examples to normalize the loss during training.

"""
from __future__ import division

import functools
import math
import tensorflow.google as tf

from REDACTED.mask_rcnn import anchors
from REDACTED.mask_rcnn import mask_rcnn_params
from REDACTED.mask_rcnn import spatial_transform
from REDACTED.mask_rcnn.object_detection import preprocessor
from REDACTED.mask_rcnn.object_detection import tf_example_decoder

MAX_NUM_INSTANCES = 100


class InputProcessor(object):
  """Base class of Input processor."""

  def __init__(self, image, output_size, short_side_image_size,
               long_side_max_image_size):
    """Initializes a new `InputProcessor`.

    This InputProcessor is tailored for MLPerf. The reference implementation
    resizes images as the following:
      1. Resize the short side to 800 pixels while keeping the aspect ratio.
      2. Clip the long side at a maximum of 1333 pixels.

    Args:
      image: The input image before processing.
      output_size: A integer tuple of the output image size in the form of
        (short_side, long_side) after calling resize_and_crop_image function.
      short_side_image_size: The image size for the short side. This is analogy
        to cfg.TRAIN.scales in the MLPerf reference model.
      long_side_max_image_size: The maximum image size for the long side. This
        is analogy to cfg.TRAIN.max_size in the MLPerf reference model.
    """
    self._image = image
    self._output_size = output_size
    self._short_side_image_size = short_side_image_size
    self._long_side_max_image_size = long_side_max_image_size
    # Parameters to control rescaling and shifting during preprocessing.
    # Image scale defines scale from original image to scaled image.
    self._image_scale = tf.constant(1.0)
    # The integer height and width of scaled image.
    self._scaled_height = tf.shape(image)[0]
    self._scaled_width = tf.shape(image)[1]
    self._ori_height = tf.shape(image)[0]
    self._ori_width = tf.shape(image)[1]

  def normalize_image(self):
    """Normalize the image to zero mean and unit variance."""
    # The image normalization is identical to Cloud TPU ResNet.
    self._image = tf.image.convert_image_dtype(self._image, dtype=tf.float32)
    offset = tf.constant([0.485, 0.456, 0.406])
    offset = tf.expand_dims(offset, axis=0)
    offset = tf.expand_dims(offset, axis=0)
    self._image -= offset

    # This is simlar to `PIXEL_MEANS` in the reference. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L909  # pylint: disable=line-too-long
    scale = tf.constant([0.229, 0.224, 0.225])
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.expand_dims(scale, axis=0)
    self._image /= scale

  def set_scale_factors_to_mlperf_reference_size(self):
    """Set the parameters to resize the image according to MLPerf reference."""
    # Compute the scale_factor using rounded scaled image size.
    height = tf.shape(self._image)[0]
    width = tf.shape(self._image)[1]
    # Recompute the accurate scale_factor using rounded scaled image size.
    # https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/utils/blob.py#L70  # pylint: disable=line-too-long
    min_image_size = tf.to_float(tf.minimum(height, width))
    max_image_size = tf.to_float(tf.maximum(height, width))
    short_side_scale = tf.to_float(self._short_side_image_size) / min_image_size
    long_side_scale = (
        tf.to_float(self._long_side_max_image_size) / max_image_size)
    image_scale = tf.minimum(short_side_scale, long_side_scale)
    scaled_height = tf.to_int32(tf.to_float(height) * image_scale)
    scaled_width = tf.to_int32(tf.to_float(width) * image_scale)
    self._image_scale = image_scale
    self._scaled_height = scaled_height
    self._scaled_width = scaled_width
    return image_scale

  def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
    """Resize input image and crop it to the self._output dimension."""
    scaled_image = tf.image.resize_images(
        self._image, [self._scaled_height, self._scaled_width], method=method)

    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    output_image = tf.cond(
        is_height_short_side,
        lambda: tf.image.pad_to_bounding_box(scaled_image, 0, 0, self._output_size[0], self._output_size[1]),  # pylint: disable=line-too-long
        lambda: tf.image.pad_to_bounding_box(scaled_image, 0, 0, self._output_size[1], self._output_size[0])  # pylint: disable=line-too-long
    )

    return output_image

  def get_image_info(self):
    """Returns image information for scaled and original height and width."""
    return tf.stack([
        tf.to_float(self._scaled_height),
        tf.to_float(self._scaled_width),
        1.0 / self._image_scale,
        tf.to_float(self._ori_height),
        tf.to_float(self._ori_width)])


class InstanceSegmentationInputProcessor(InputProcessor):
  """Input processor for object detection."""

  def __init__(self, image, output_size, short_side_image_size,
               long_side_max_image_size, boxes=None, classes=None, masks=None):
    InputProcessor.__init__(self, image, output_size, short_side_image_size,
                            long_side_max_image_size)
    self._boxes = boxes
    self._classes = classes
    self._masks = masks

  def random_horizontal_flip(self):
    """Randomly flip input image and bounding boxes."""
    self._image, self._boxes, self._masks = preprocessor.random_horizontal_flip(
        self._image, boxes=self._boxes, masks=self._masks)

  def clip_boxes(self, boxes):
    """Clip boxes to fit in an image."""
    boxes = tf.where(tf.less(boxes, 0), tf.zeros_like(boxes), boxes)
    is_height_short_side = tf.less(self._scaled_height, self._scaled_width)
    bound = tf.where(
        is_height_short_side,
        tf.convert_to_tensor(
            [self._output_size[0] - 1, self._output_size[1] - 1] * 2,
            dtype=tf.float32),
        tf.convert_to_tensor(
            [self._output_size[1] - 1, self._output_size[0] - 1] * 2,
            dtype=tf.float32))
    boxes = tf.where(
        tf.greater(boxes, bound), bound * tf.ones_like(boxes), boxes)
    return boxes

  def resize_and_crop_boxes(self):
    """Resize boxes and crop it to the self._output dimension."""
    boxlist = preprocessor.box_list.BoxList(self._boxes)
    boxes = preprocessor.box_list_scale(
        boxlist, self._scaled_height, self._scaled_width).get()
    # Clip the boxes.
    boxes = self.clip_boxes(boxes)
    # Filter out ground truth boxes that are all zeros and corresponding classes
    # and masks.
    indices = tf.where(tf.not_equal(tf.reduce_sum(boxes, axis=1), 0))
    boxes = tf.gather_nd(boxes, indices)
    classes = tf.gather_nd(self._classes, indices)
    self._masks = tf.gather_nd(self._masks, indices)
    return boxes, classes

  def crop_gt_masks(self, gt_mask_size):
    """Crops the ground truth binary masks and resize to fixed-size masks."""
    num_boxes = tf.shape(self._boxes)[0]
    num_masks = tf.shape(self._masks)[0]
    assert_length = tf.Assert(
        tf.equal(num_boxes, num_masks), [num_masks])

    def padded_bounding_box_fn():
      return tf.reshape(self._masks, [-1, self._ori_height, self._ori_width, 1])

    def zeroed_box_fn():
      return tf.zeros([0, self._ori_height, self._ori_width, 1])

    num_masks = tf.shape(self._masks)[0]
    # Check if there is any instance in this image or not.
    scaled_masks = tf.cond(num_masks > 0, padded_bounding_box_fn, zeroed_box_fn)
    with tf.control_dependencies([assert_length]):
      cropped_gt_masks = tf.image.crop_and_resize(
          image=scaled_masks, boxes=self._boxes,
          box_ind=tf.range(num_masks, dtype=tf.int32),
          crop_size=[gt_mask_size, gt_mask_size],
          method='bilinear')[:, :, :, 0]
    cropped_gt_masks = tf.pad(
        cropped_gt_masks, paddings=tf.constant([[0, 0,], [2, 2,], [2, 2]]),
        mode='CONSTANT', constant_values=0.)
    return cropped_gt_masks


def pad_to_fixed_size(data, pad_value, output_shape):
  """Pad data to a fixed length at the first dimension.

  Args:
    data: Tensor to be padded to output_shape.
    pad_value: A constant value assigned to the paddings.
    output_shape: The output shape of a 2D tensor.

  Returns:
    The Padded tensor with output_shape [max_num_instances, dimension].
  """
  max_num_instances = output_shape[0]
  dimension = output_shape[1]
  data = tf.reshape(data, [-1, dimension])
  num_instances = tf.shape(data)[0]
  assert_length = tf.Assert(
      tf.less_equal(num_instances, max_num_instances), [num_instances])
  with tf.control_dependencies([assert_length]):
    pad_length = max_num_instances - num_instances
  paddings = pad_value * tf.ones([pad_length, dimension])
  padded_data = tf.concat([data, paddings], axis=0)
  padded_data = tf.reshape(padded_data, output_shape)
  return padded_data


class InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, mode=tf.estimator.ModeKeys.TRAIN,
               use_fake_data=False, distributed_eval=False):
    if mode not in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT]:
      raise ValueError('InputReader supports only TRAIN or PREDICT modes.')

    self._file_pattern = file_pattern
    self._max_num_instances = MAX_NUM_INSTANCES
    self._mode = mode
    self._use_fake_data = use_fake_data
    self._distributed_eval = distributed_eval

  def __call__(self, params, num_examples=0):
    image_size = params['image_size']
    input_anchors = anchors.Anchors(
        params['min_level'], params['max_level'], params['num_scales'],
        params['aspect_ratios'], params['anchor_scale'], image_size)
    anchor_labeler = anchors.AnchorLabeler(
        input_anchors, params['num_classes'], params['rpn_positive_overlap'],
        params['rpn_negative_overlap'], params['rpn_batch_size_per_im'],
        params['rpn_fg_fraction'])

    height_long_side_image_size = image_size[::-1]
    height_long_side_input_anchors = anchors.Anchors(
        params['min_level'], params['max_level'], params['num_scales'],
        params['aspect_ratios'], params['anchor_scale'],
        height_long_side_image_size)
    height_long_side_anchor_labeler = anchors.AnchorLabeler(
        height_long_side_input_anchors, params['num_classes'],
        params['rpn_positive_overlap'], params['rpn_negative_overlap'],
        params['rpn_batch_size_per_im'], params['rpn_fg_fraction'])

    example_decoder = tf_example_decoder.TfExampleDecoder(
        use_instance_mask=True)

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        features: A dictionary that contains the image and auxiliary
          information. The following describes {key: value} pairs in the
          dictionary.
          image: An image tensor that is preprocessed to have normalized value
            and fixed dimension [image_size, image_size, 3]
          image_info: Image information that includes the original height and
            width, the scale of the processed image to the original image, and
            the scaled height and width.
          source_ids: Source image id. Default value -1 if the source id is
            empty in the groundtruth annotation.
        labels: (only for training) A dictionary that contains groundtruth
          labels. The following describes {key: value} pairs in the dictionary.
          score_targets_dict: An ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors]. The height_l and width_l
            represent the dimension of objectiveness score at l-th level.
          box_targets_dict: An ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
          gt_boxes: Groundtruth bounding box annotations. The box is represented
             in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
             fixed dimension [self._max_num_instances, 4].
          gt_classes: Groundtruth classes annotations. The tennsor is padded
            with -1 to the fixed dimension [self._max_num_instances].
          cropped_gt_masks: Groundtruth masks cropped by the bounding box and
            resized to a fixed size determined by params['gt_mask_size']
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)

        image = data['image']
        source_id = data['source_id']
        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
          input_processor = InstanceSegmentationInputProcessor(
              image, image_size, params['short_side_image_size'],
              params['long_side_max_image_size'])
          input_processor.normalize_image()
          input_processor.set_scale_factors_to_mlperf_reference_size()
          image = input_processor.resize_and_crop_image()
          if params['use_bfloat16']:
            image = tf.cast(image, dtype=tf.bfloat16)

          image_info = input_processor.get_image_info()
          return {'images': image, 'image_info': image_info,
                  'source_ids': source_id}

        # The following part is for training.
        instance_masks = data['groundtruth_instance_masks']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        if not params['use_category']:
          classes = tf.cast(tf.greater(classes, 0), dtype=tf.float32)

        if (params['skip_crowd_during_training'] and
            self._mode == tf.estimator.ModeKeys.TRAIN):
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)
          instance_masks = tf.gather_nd(instance_masks, indices)

        input_processor = InstanceSegmentationInputProcessor(
            image, image_size, params['short_side_image_size'],
            params['long_side_max_image_size'], boxes, classes,
            instance_masks)
        input_processor.normalize_image()
        if params['input_rand_hflip']:
          input_processor.random_horizontal_flip()

        input_processor.set_scale_factors_to_mlperf_reference_size()
        image = input_processor.resize_and_crop_image()

        boxes, classes = input_processor.resize_and_crop_boxes()
        cropped_gt_masks = input_processor.crop_gt_masks(
            params['gt_mask_size'])

        image_info = input_processor.get_image_info()
        # Assign anchors.
        is_height_short_side = tf.less(image_info[3], image_info[4])
        score_targets, box_targets = tf.cond(
            is_height_short_side,
            lambda: anchor_labeler.label_anchors(boxes, classes),
            lambda: height_long_side_anchor_labeler.label_anchors(boxes, classes))  # pylint: disable=line-too-long

        # Pad groundtruth data.
        boxes *= image_info[2]
        boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
        classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
        # Pads cropped_gt_masks.
        cropped_gt_masks = tf.reshape(
            cropped_gt_masks, [-1, (params['gt_mask_size'] + 4) ** 2])
        cropped_gt_masks = pad_to_fixed_size(
            cropped_gt_masks, -1,
            [self._max_num_instances, (params['gt_mask_size'] + 4) ** 2])
        cropped_gt_masks = tf.reshape(
            cropped_gt_masks,
            [self._max_num_instances, params['gt_mask_size'] + 4,
             params['gt_mask_size'] + 4])
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)

        features = {}
        features['images'] = image
        features['image_info'] = image_info
        features['source_ids'] = source_id

        labels = {}
        for level in range(params['min_level'], params['max_level'] + 1):
          labels['score_targets_%d' % level] = score_targets[level]
          labels['box_targets_%d' % level] = box_targets[level]
        labels['gt_boxes'] = boxes
        labels['gt_classes'] = classes
        labels['cropped_gt_masks'] = cropped_gt_masks
        return features, labels

    batch_size = params['batch_size'] if 'batch_size' in params else 1
    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)
    if self._mode == tf.estimator.ModeKeys.TRAIN:
      # shard and shuffle the image files so each shard has distinctive and
      # random set of images.
      # To improve model convergence under large number of hosts, multiple hosts
      # may share a same dataset shard. This allows a host to get more training
      # images.
      if 'dataset_num_shards' in params:
        train_actual_num_shards = int(params['dataset_num_shards'] //
                                      params['hosts_per_dataset_shard'])
        dataset = dataset.shard(
            train_actual_num_shards,
            params['dataset_index'] // params['hosts_per_dataset_shard'])
        if not self._use_fake_data:
          dataset = dataset.shuffle(tf.to_int64(256 // train_actual_num_shards))

    if self._distributed_eval:
      dataset = dataset.shard(params['dataset_num_shards'],
                              params['dataset_index'])

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(filename).prefetch(1)
      return dataset

    shuffle_data = (
        self._mode == tf.estimator.ModeKeys.TRAIN) and not self._use_fake_data
    concurrent_files = 16
    dataset = dataset.interleave(
        _prefetch_dataset,
        cycle_length=concurrent_files,
        block_length=1,
        num_parallel_calls=concurrent_files)

    if shuffle_data:
      # Cache the raw images and shuffle them with a resonably large buffer.
      dataset = dataset.cache().shuffle(
          params['shuffle_buffer_size'],
          reshuffle_each_iteration=True).repeat()

    # Parse the fetched records to input tensors for model function.
    dataset = dataset.map(_dataset_parser, num_parallel_calls=64)

    def horizontal_image(*args):
      image_info = args[0]['image_info']
      return tf.less(image_info[3], image_info[4])

    def vertical_image(*args):
      return tf.logical_not(horizontal_image(*args))

    # Pad dataset to the desired size and mark if the dataset is padding.
    # During PREDICT, if batch_size_per_shard * num_shards > 5000, the
    # original dataset size won't be evenly divisible by the number of shards.
    # Note that 5000 is the number of eval samples in COCO dataset. In this
    # case, the eval dataset will take (batch_per_shard * num_shards - 5000)
    # samples from the original dataset and mark those extra samples as
    # `is_padding` and the original data as `is_not_padding`. This ensures
    # correctness of evaluation on only 5000 samples.
    # Appends the dataset padding to the original dataset (only in PREDICT).
    if (self._mode == tf.estimator.ModeKeys.PREDICT and
        num_examples > params['eval_samples']):
      def _mark_is_padding(features):
        features[mask_rcnn_params.IS_PADDED] = tf.constant(
            True, dtype=tf.bool, shape=[1])
        return features

      def _mark_is_not_padding(features):
        features[mask_rcnn_params.IS_PADDED] = tf.constant(
            False, dtype=tf.bool, shape=[1])
        return features
      dataset_padding = dataset
      # padd equal number of horizontal and vertical images and interleave them.
      pad_size = int(math.ceil(num_examples - params['eval_samples']))
      dataset_padding_hor = dataset_padding.filter(horizontal_image).map(
          _mark_is_padding).take(pad_size)
      dataset_padding_ver = dataset_padding.filter(vertical_image).map(
          _mark_is_padding).take(pad_size)
      interleaved_dataset_padding = tf.data.experimental.choose_from_datasets(
          [dataset_padding_hor, dataset_padding_ver],
          tf.data.Dataset.range(2).repeat(pad_size))
      if self._distributed_eval:
        dataset = dataset.map(_mark_is_not_padding).take(
            int(
                math.ceil(params['eval_samples'] /
                          params['dataset_num_shards'])))
      else:
        dataset = dataset.map(_mark_is_not_padding).take(params['eval_samples'])
      dataset = dataset.concatenate(interleaved_dataset_padding)

    def key_func(*args):
      return tf.cast(horizontal_image(*args), dtype=tf.int64)

    def reduce_func(unused_key, dataset):
      return dataset.batch(batch_size, drop_remainder=True)

    dataset = dataset.apply(
        tf.data.experimental.group_by_window(
            key_func=key_func,
            reduce_func=reduce_func,
            window_size=(params['batch_size'] * params['replicas_per_host'])))

    dataset = dataset.map(
        functools.partial(self._transform_images, params),
        num_parallel_calls=16)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if (self._mode == tf.estimator.ModeKeys.TRAIN and
        num_examples > 0):
      dataset = dataset.take(num_examples)
    # Make eval dataset repeat to get rid of eval dataset init per epoch.
    if self._distributed_eval:
      dataset = dataset.take(
          int(num_examples / params['dataset_num_shards'] /
              params['batch_size'])).cache().repeat()
    if self._use_fake_data:
      # Turn this dataset into a semi-fake dataset which always loop at the
      # first batch. This reduces variance in performance and is useful in
      # testing.
      dataset = dataset.take(1).cache().repeat()

    deterministic = (not shuffle_data)
    options = tf.data.Options()
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = deterministic
    dataset = dataset.with_options(options)

    return dataset

  def _transform_images(self, params, features, labels=None):
    """Transforms images."""

    images = features['images']
    batch_size, _, _, c = images.get_shape().as_list()
    if params['conv0_space_to_depth_block_size'] != 0:
      # Transforms (space-to-depth) images for TPU performance.

      def _fused_transform(images, image_size):
        return spatial_transform.fused_transpose_and_space_to_depth(
            images, image_size, params['conv0_space_to_depth_block_size'],
            params['transpose_input'])

      images = tf.cond(
          tf.less(features['image_info'][0, 3], features['image_info'][0, 4]),
          lambda: _fused_transform(images, params['image_size']),
          lambda: _fused_transform(images, params['image_size'][::-1]))

    else:
      # Transposes images for TPU performance.
      image_area = params['image_size'][0] * params['image_size'][1]
      if params['transpose_input']:
        images = tf.transpose(images, [1, 2, 0, 3])
        # Flattens spatial dimensions so that the image tensor has a static
        # shape.
        images = tf.reshape(images, [image_area, batch_size, c])
      else:
        images = tf.reshape(images, [batch_size, image_area, c])

    if params['use_bfloat16']:
      images = tf.cast(images, dtype=tf.bfloat16)

    features['images'] = images

    if labels is not None:
      return features, labels
    else:
      return features, tf.zeros([batch_size])
