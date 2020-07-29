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
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from absl import flags
import jax
import tensorflow as tf

FLAGS = flags.FLAGS
TRAIN_IMAGES = 1281167
EVAL_IMAGES = 50000
IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


flags.DEFINE_string(
    'data_dir', default='/readahead/128M/placer/prod/home/distbelief/imagenet-tensorflow/imagenet-2012-tfrecord',
    help='Directory to load data from.')


def load_split(batch_size, train, dtype, image_format, space_to_depth,
               cache_uncompressed, image_size=IMAGE_SIZE, reshape_to_r1=False,
               shuffle_size=16384):
  """Returns the input_fn."""

  def dataset_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.io.parse_single_example(
        value, {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64, 0)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1

    def preprocess_fn():
      """Preprocess the image."""
      shape = tf.image.extract_jpeg_shape(image_bytes)
      if train:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_bytes),
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.05, 1.0),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack(
            [offset_y, offset_x, target_height, target_width])
      else:
        crop_size = tf.cast(
            ((image_size / (image_size + CROP_PADDING)) *
             tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)), tf.int32)
        offset_y, offset_x = [
            ((shape[i] - crop_size) + 1) // 2 for i in range(2)
        ]
        crop_window = tf.stack([offset_y, offset_x, crop_size, crop_size])

      image = tf.image.decode_and_crop_jpeg(
          image_bytes, crop_window, channels=3)
      image = tf.image.resize(
          [image], [image_size, image_size], method='bicubic')[0]
      if train:
        image = tf.image.random_flip_left_right(image)
      image = tf.reshape(image, [image_size, image_size, 3])
      return tf.image.convert_image_dtype(image, dtype)

    empty_example = tf.zeros([image_size, image_size, 3], dtype)
    return tf.cond(label < 0, lambda: empty_example, preprocess_fn), label

  def cached_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.io.parse_single_example(
        value, {
            'image/encoded': tf.io.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64, 0)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    image_bytes = tf.io.decode_jpeg(image_bytes, channels=3)
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1
    return image_bytes, label

  def crop_image(image_bytes, label):
    """Preprocess the image."""
    shape = tf.shape(image_bytes)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.08, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                          target_height, target_width)
    image = tf.image.resize(
        [image], [image_size, image_size], method='bicubic')[0]
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [image_size, image_size, 3])
    return tf.image.convert_image_dtype(image, dtype), label

  def set_shapes(images, labels):
    """Statically set the batch_size dimension."""
    if image_format == 'NHWC':
      shape = [batch_size, None, None, None]
    elif image_format == 'HWCN':
      shape = [None, None, None, batch_size]
    elif image_format == 'HWNC':
      shape = [None, None, batch_size, None]
    else:
      raise ValueError('unknown format: {}'.format(image_format))
    images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
    if reshape_to_r1:
      images = tf.reshape(images, [-1])
    labels.set_shape([batch_size])
    return images, labels

  index = jax.host_id()
  num_hosts = jax.host_count()
  replicas_per_host = jax.local_device_count()
  steps = math.ceil(EVAL_IMAGES / (batch_size * replicas_per_host * num_hosts))
  num_dataset_per_shard = max(1, int(steps * batch_size * replicas_per_host))
  padded_dataset = tf.data.Dataset.from_tensors(
      tf.constant(
          tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'image/class/label':
                          tf.train.Feature(
                              int64_list=tf.train.Int64List(value=[0])),
                      'image/encoded':
                          tf.train.Feature(
                              bytes_list=tf.train.BytesList(
                                  value=[str.encode('')]))
                  })).SerializeToString(),
          dtype=tf.string)).repeat(num_dataset_per_shard)

  if FLAGS.data_dir is None:
    dataset = padded_dataset.repeat().map(dataset_parser, 64)
  else:
    file_pattern = os.path.join(FLAGS.data_dir,
                                'train-*' if train else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.shard(num_hosts, index)
    concurrent_files = min(10, 1024 // num_hosts)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, concurrent_files, 1, concurrent_files)

    if train:
      if cache_uncompressed:
        dataset = dataset.map(cached_parser, 64).cache()
        dataset = dataset.shuffle(
            shuffle_size, reshuffle_each_iteration=True).repeat()
        dataset = dataset.map(crop_image, 64)
      else:
        dataset = dataset.cache()  # cache compressed JPEGs instead
        dataset = dataset.shuffle(
            shuffle_size, reshuffle_each_iteration=True).repeat()
        dataset = dataset.map(dataset_parser, 64)
    else:
      dataset = dataset.concatenate(padded_dataset).take(num_dataset_per_shard)
      dataset = dataset.map(dataset_parser, 64)
  dataset = dataset.batch(batch_size, True)

  if space_to_depth:
    dataset = dataset.map(
        lambda images, labels: (tf.nn.space_to_depth(images, 2), labels), 64)
  # Transpose for performance on TPU
  if image_format == 'HWCN':
    transpose_array = [1, 2, 3, 0]
  elif image_format == 'HWNC':
    transpose_array = [1, 2, 0, 3]
  if image_format != 'NCHW':
    dataset = dataset.map(
        lambda imgs, labels: (tf.transpose(imgs, transpose_array), labels), 64)
  dataset = dataset.map(set_shapes, 64)
  if not train:
    dataset = dataset.cache().repeat()
  dataset = dataset.prefetch(100)

  options = tf.data.Options()
  options.experimental_deterministic = False
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_threading.private_threadpool_size = 48
  dataset = dataset.with_options(options)
  return dataset
