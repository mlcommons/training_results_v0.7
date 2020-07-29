# Lint as: python3
"""SSD input pipeline modified for JAX.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from absl import flags

import tensorflow.compat.v2 as tf


from REDACTED.cloud_tpu.models.retinanet.object_detection import box_list
from REDACTED.tensorflow_models.mlperf.models.rough.ssd import dataloader
from REDACTED.tensorflow_models.mlperf.models.rough.ssd_jax import ssd_constants

FLAGS = flags.FLAGS


# Original num_boxes_static function does not work when eagerly called from jax.
# Patch the function to get around this issue.
def num_boxes_static(self):
  return self.data['boxes'].get_shape()[0]
box_list.BoxList.num_boxes_static = num_boxes_static


def fused_transpose_and_space_to_depth(
    images,
    block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
    transpose_input=True):
  """Fuses space-to-depth and transpose.

  Space-to-depth performs the following permutation, which is equivalent to
  tf.nn.space_to_depth.

  images = tf.reshape(images, [batch, h // block_size, block_size,
                               w // block_size, block_size, c])
  images = tf.transpose(images, [0, 1, 3, 2, 4, 5])
  images = tf.reshape(images, [batch, h // block_size, w // block_size,
                               c * (block_size ** 2)])

  Args:
    images: A tensor with a shape of [batch_size, h, w, c] as the images. The h
      and w can be dynamic sizes.
    block_size: A integer for space-to-depth block size.
    transpose_input: A boolean to indicate if the images tensor should be
      transposed.

  Returns:
    A transformed images tensor.

  """
  local_num_replicas, batch_size, h, w, c = images.get_shape().as_list()
  images = tf.reshape(images, [
      local_num_replicas, batch_size, h // block_size, block_size,
      w // block_size, block_size, c
  ])
  if transpose_input:
    if batch_size > 8:
      # TODO(b/153450272): below transpose might be slow. Try numpy transpose.
      # HWCN
      images = tf.transpose(images, [0, 2, 4, 3, 5, 6, 1])
      images = tf.reshape(images, [
          local_num_replicas, h // block_size, w // block_size, c *
          (block_size**2), batch_size
      ])
    else:
      # TODO(b/153450272): below transpose might be slow. Try numpy transpose.
      # HWNC
      images = tf.transpose(images, [0, 2, 4, 1, 3, 5, 6])
      images = tf.reshape(images, [
          local_num_replicas, h // block_size, w // block_size, batch_size, c *
          (block_size**2)
      ])
  else:
    # TODO(b/153450272): below transpose might be slow. Try numpy transpose.
    images = tf.transpose(images, [0, 1, 2, 4, 3, 5, 6])
    images = tf.reshape(images, [
        local_num_replicas, batch_size, h // block_size, w // block_size, c *
        (block_size**2)
    ])
  return images


def get_spmd_image_padding_size(params, original_image_shape):
  """Returns the padding size to make it divisible by num_parts.

  SPMD partitions along the height dimension, this function returns the padding
  size along this dim.

  Args:
    params: Dictionary for SSD parameters.
    original_image_shape: 3D Image shape (H, W, C). If conv_space_to_depth
      is applied, the image shape after this optimization.

  Returns:
    Integer denoting the padding to add second dimension.
  """
  padding_size = 0
  if 'num_partitions' in params and params['num_partitions'] > 1:
    # SPMD is not supported with transpose input.
    assert not params['transpose_input']
    assert len(original_image_shape) == 3
    height_dim_ind = 0
    part_dim_size = original_image_shape[height_dim_ind]
    left_over = part_dim_size % params['num_partitions']
    if left_over:
      padding_size = params['num_partitions'] - left_over
  return padding_size


def ssd_input_pipeline(params, file_pattern, is_training=False,
                       use_fake_data=False,
                       transpose_input=False, distributed_eval=False,
                       count=-1, host_batch_size=-1):
  """Sets up and returns the training dataset for ssd.

  Args:
    params: Dictionary containing SSD parameters.
    file_pattern: File pattern to match input tf records files.
    is_training: True if training.
    use_fake_data: True if test with fake.
    transpose_input: True to apply transpose input trick.
    distributed_eval: True if eval is distributed.
    count: Number of dataset instances that will be iterated over.
    host_batch_size: batch size per host. If not provided, it will look for
      params['host_batch_size'].

  Returns:
    A tf.dataset object. Eatch batch contains a tuple corresponding to images,
      and labels. Each image is a 5D tensor with dimensions
      (local_num_replicas, device_batch_size, image_size, image_size, #channels)
      Labels contain boxes, classes and num_matched_boxes, whose dimensions
      will be
        * (local_num_replicas, device_batch_size, num_boxes, 4)
        * (local_num_replicas, device_batch_size, num_boxes)
        * (local_num_replicas, device_batch_size),
      respectively.

  """
  def pad_images_if_uneven(images):
    # If num partitions is larger than 1, then pad the input so that
    # it passes JAX's ragged partition checking.
    # Otherwise it throws an error when trying to partition second
    # dimension (150) into 4 parts.
    # Image shape [local_device, B, H, W, C] --> # [B, H, W, C]
    padding_size = get_spmd_image_padding_size(params, images.shape[2:])
    if padding_size:
      padding_spec = tf.constant([[0, 0],
                                  [0, 0],
                                  [0, padding_size],
                                  [0, 0],
                                  [0, 0]])
      images = tf.pad(images, padding_spec)
    return images

  def get_new_shape(tensor, num_local_replicas):
    new_shape = [num_local_replicas] + tensor.shape.as_list()
    new_shape[1] = new_shape[1] // num_local_replicas
    return new_shape

  def split_a_tensor_or_a_dict(tensor_or_dict, num_local_replicas):
    if isinstance(tensor_or_dict, dict):
      for k, v in tensor_or_dict.items():
        tensor_or_dict[k] = tf.reshape(v, get_new_shape(v, num_local_replicas))
    else:
      tensor_or_dict = tf.reshape(tensor_or_dict,
                                  get_new_shape(tensor_or_dict,
                                                num_local_replicas))
    return tensor_or_dict

  def shard_train_batch(images, labels):
    """Converts 4D input into 5D input where first dimension denotes cores."""

    boxes = labels[ssd_constants.BOXES]
    classes = labels[ssd_constants.CLASSES]
    num_matched_boxes = labels[ssd_constants.NUM_MATCHED_BOXES]

    local_num_replicas = params['local_num_replicas']

    images = split_a_tensor_or_a_dict(images, local_num_replicas)
    boxes = split_a_tensor_or_a_dict(boxes, local_num_replicas)
    classes = split_a_tensor_or_a_dict(classes, local_num_replicas)

    num_matched_boxes = split_a_tensor_or_a_dict(
        tf.reshape(num_matched_boxes, [-1]), local_num_replicas)

    if params['conv0_space_to_depth']:

      def _space_to_depth_training_fn(images, labels):
        images = fused_transpose_and_space_to_depth(
            images,
            block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            transpose_input=transpose_input)
        if transpose_input:
          labels = tf.transpose(labels, [0, 2, 3, 1])
        images = pad_images_if_uneven(images)
        return images, labels
      images, boxes = _space_to_depth_training_fn(images, boxes)
    elif transpose_input:

      # numpy's 5D tranpose is faster than tf 5D transpose.
      # pylint: disable=protected-access
      def np_transpose_bs_gt_8(x):
        return tf.convert_to_tensor(x._numpy().transpose([0, 2, 3, 4, 1]))
      def np_transpose_bs_le_8(x):
        return tf.convert_to_tensor(x._numpy().transpose([0, 2, 3, 1, 4]))

      # pylint: enable=protected-access

      if host_batch_size // params['local_num_replicas'] > 8:
        images = tf.py_function(np_transpose_bs_gt_8, [images],
                                Tout=images.dtype)
      else:
        images = tf.py_function(np_transpose_bs_le_8, [images],
                                Tout=images.dtype)
      # Use tf tranpose on 4D tensor.
      boxes = tf.transpose(boxes, [0, 2, 3, 1])

    return (images, {ssd_constants.BOXES: boxes,
                     ssd_constants.CLASSES: classes,
                     ssd_constants.NUM_MATCHED_BOXES: num_matched_boxes})

  def shard_eval_batch(images, labels):
    """Converts 4D input into 5D input where first dimension denotes cores."""

    boxes = labels[ssd_constants.BOXES]
    classes = labels[ssd_constants.CLASSES]
    source_id = labels[ssd_constants.SOURCE_ID]
    raw_shape = labels[ssd_constants.RAW_SHAPE]
    local_num_replicas = params['local_num_replicas']
    image_size = ssd_constants.IMAGE_SIZE
    num_boxes = ssd_constants.MAX_NUM_EVAL_BOXES
    num_corners = 4
    out_labels = {}

    if count > params['eval_samples']:
      out_labels[ssd_constants.IS_PADDED] = labels[ssd_constants.IS_PADDED]

    images = tf.reshape(images,
                        [local_num_replicas, -1, image_size, image_size, 3])
    boxes = tf.reshape(boxes, [local_num_replicas, -1,
                               num_boxes, num_corners])
    classes = tf.reshape(classes, [local_num_replicas, -1, num_boxes])

    source_id = tf.reshape(source_id, [local_num_replicas, -1])
    raw_shape = tf.reshape(raw_shape, [local_num_replicas, -1, 3])

    if count > params['eval_samples']:
      out_labels[ssd_constants.IS_PADDED] = tf.reshape(
          out_labels[ssd_constants.IS_PADDED], [local_num_replicas, -1])

    if params['conv0_space_to_depth']:
      if transpose_input:
        raise ValueError('Transpose input for eval is not implemented yet.')

      def _space_to_depth_training_fn(images, labels):
        images = fused_transpose_and_space_to_depth(
            images,
            block_size=ssd_constants.SPACE_TO_DEPTH_BLOCK_SIZE,
            transpose_input=transpose_input)
        images = pad_images_if_uneven(images)
        return images, labels
      images, boxes = _space_to_depth_training_fn(images, boxes)
    elif transpose_input:
      raise ValueError('Transpose input for eval is not implemented yet.')

    out_labels.update({ssd_constants.BOXES: boxes,
                       ssd_constants.CLASSES: classes,
                       ssd_constants.SOURCE_ID: source_id,
                       ssd_constants.RAW_SHAPE: raw_shape})
    return (images, out_labels)

  if host_batch_size == -1:
    host_batch_size = params['host_batch_size']

  batch_params = {
      'batch_size': host_batch_size,
      'dataset_num_shards': params['dataset_num_shards'],
      'dataset_index': params['dataset_index']
  }
  # Do not let SSDInputReader to transpose, as we will need to transpose it back
  # Do not let SSDInputReader do space to depth
  saved_conv0_space_to_depth_val = params['conv0_space_to_depth']
  params['conv0_space_to_depth'] = False
  ssd_input_ds = dataloader.SSDInputReader(
      file_pattern, transpose_input=False,
      is_training=is_training, use_fake_data=use_fake_data,
      distributed_eval=distributed_eval, count=count,
      params=params)(batch_params)
  params['conv0_space_to_depth'] = saved_conv0_space_to_depth_val
  if is_training:
    return ssd_input_ds.map(
        shard_train_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
  else:
    return ssd_input_ds.map(
        shard_eval_batch,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat()
