#!/usr/bin/env python
# TODO(ahmadki): add segmentation and area to TFRecord
import os
import json
import random
import argparse
from subprocess import call
from collections import defaultdict
from itertools import accumulate

from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TF
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare TFRecord dataset from COCO')
    parser.add_argument('-i', '--images-dir', type=str,
                        help='images root dir')
    parser.add_argument('-a', '--annotations-file', type=str,
                        help='COCO annotation file')
    parser.add_argument('-n', '--num-shards', type=int, default=1,
                        help='Split the dataset to --num-shards')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file name. shard idx and .tfrecord will be appended to the fname')
    parser.add_argument('--ltrb', action='store_true',
                        help='If true, bboxes are saved as [left, top, right, bottom], else [x, y, width, height]')
    parser.add_argument('--ratio', action='store_true',
                        help='If true, bboxes are saved as ratio w.r.t. to the image width and height.')
    parser.add_argument('--skip-empty', action='store_true',
                        help='If true, samples with no object instances in them will not be saved')
    parser.add_argument('--size-threshold', type=float, default=0.0,
                        help='Bounding boxes with width or height under this value will not be saved.')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Seed to shuffle images before crteating the TFRecord')
    args = parser.parse_args()
    return args


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def int64_feature_list(values):
  return tf.train.FeatureList(feature=[int64_feature(v) for v in values])

def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def bytes_feature_list(values):
  return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])

def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def float_feature_list(values):
  return tf.train.FeatureList(feature=[float_feature(v) for v in values])


def coco_to_tf_example(images_dir, image_dict, annotations, category_dict, labels_mapper,
                       ltrb, ratio, size_threshold):
    image_id = image_dict['id']
    image_fname = image_dict['file_name']
    image_height = image_dict['height']
    image_width = image_dict['width']
    with open(os.path.join(images_dir, image_fname), 'rb') as fp:
        encoded_image = fp.read()

    object_labels = []
    object_names = []
    bboxes = []
    is_crowd = []
    skipped_bboxes = 0
    for annotation in annotations:
        # BBoxes
        (x, y, width, height) = tuple(annotation['bbox'])
        if width < size_threshold or height < size_threshold:
            skipped_bboxes += 1
            continue

        bbox0 = float(x)
        bbox1 = float(y)
        bbox2 = float(width)
        bbox3 = float(height)

        if ltrb:
            bbox0 = bbox0          # x_min
            bbox1 = bbox1          # y_min
            bbox2 = bbox0 + bbox2  # x_max
            bbox3 = bbox1 + bbox3  # y_max
            # clip values to [0, image_width/image_height]
            bbox0 = max(0, min(bbox0, image_width))
            bbox1 = max(0, min(bbox1, image_height))
            bbox2 = max(0, min(bbox2, image_width))
            bbox3 = max(0, min(bbox3, image_height))
        if ratio:
            bbox0 = bbox0 / image_width
            bbox1 = bbox1 / image_height
            bbox2 = bbox2 / image_width
            bbox3 = bbox3 / image_height

        bboxes.extend([bbox0, bbox1, bbox2, bbox3])

        object_label = annotation['category_id']
        is_crowd.append(annotation['iscrowd'])
        object_labels.append(labels_mapper[annotation['category_id']])
        object_names.append(category_dict[str(object_label)]['name'].encode('utf8'))

    num_bboxes = len(bboxes)
    feature_dict = {
        'image/fname':           bytes_feature(image_fname.encode('utf8')),
        'image/encoded':         bytes_feature(encoded_image),
        'image/id':              int64_feature(image_id),
        'image/shape':           int64_feature([image_height, image_width, 3]),
        'image/object/label':    int64_feature(object_labels),
        'image/object/name':     bytes_feature(object_names),
        'image/object/bbox':     float_feature(bboxes),
        'image/object/is_crowd': int64_feature(is_crowd),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, num_bboxes, skipped_bboxes


def coco_to_tfrecord(images_dir, annotations_file, output, num_shards,
                     ratio, ltrb, size_threshold, skip_empty, seed=None):
    deleted_labels = [0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
    labels_mapper = [1]*92
    for i in deleted_labels:
        labels_mapper[i] = 0
    labels_mapper = list(accumulate(labels_mapper))

    print("Loading annotation file...")
    with open(annotations_file) as fp:
        annotations_dict = json.load(fp)
    coco_images = annotations_dict['images']
    coco_annotations = annotations_dict['annotations']
    coco_categories = annotations_dict['categories']

    print("Preparing internal data structures...")
    categories_dict = {}
    for category in tqdm(coco_categories, desc='Building categories dictionary'):
        categories_dict[str(category['id'])] = category

    annotations_dict = defaultdict(list)
    for annotation in tqdm(coco_annotations, desc='Building annotations dictionary'):
        image_id = str(annotation['image_id'])
        annotations_dict[image_id].append(annotation)

    images_dict = {}
    for image in tqdm(coco_images, desc='Building images dictionary'):
        image_id = str(image['id'])
        images_dict[image_id] = image
        images_dict[image_id]['annotations'] = annotations_dict[image_id]

    image_ids = list(images_dict.keys())
    if seed:
        random.seed(seed)
        random.shuffle(image_ids)

    tfrecord_writers = []
    for i in range(num_shards):
        fname = '{}.{}_of_{}.tfrecord'.format(output, i+1, num_shards)
        tfrecord_writers.append(tf.io.TFRecordWriter(fname))

    skipped_bboxes = 0
    num_empty_images = 0
    num_written_bboxes = 0
    num_written_images = 0
    for image_id in tqdm(image_ids, desc='Writing dataset to TFRecords'):
        annotations = annotations_dict[image_id]
        tf_example, num_bboxes, skipped_bboxes = coco_to_tf_example(images_dir=images_dir,
                                                                    image_dict=images_dict[str(image_id)],
                                                                    annotations=annotations,
                                                                    category_dict=categories_dict,
                                                                    labels_mapper=labels_mapper,
                                                                    ltrb=ltrb,
                                                                    ratio=ratio,
                                                                    size_threshold=size_threshold)
        skipped_bboxes += skipped_bboxes
        if num_bboxes==0:
            num_empty_images += 1
            if skip_empty:
                continue
        tfrecord_writers[i%num_shards].write(tf_example.SerializeToString())
        num_written_images += 1
        num_written_bboxes += num_bboxes

    for i in range(num_shards):
        tfrecord_writers[i].close()

    for i in tqdm(range(num_shards), 'Creating DALI indexes'):
        tfrecord_fname = '{}.{}_of_{}.tfrecord'.format(output, i+1, num_shards)
        idx_fname = '{}.{}_of_{}.idx'.format(output, i+1, num_shards)
        call(["tfrecord2idx", tfrecord_fname, idx_fname])

    print("Dataset size: {}".format(len(image_ids)))
    print("Total skipped bboxes(--size-threshold): {}".format(skipped_bboxes))
    print("Empty images (with no bboxes): {}".format(num_empty_images))
    print("Total images saved: {}".format(num_written_images))
    print("Total bboxes saved: {}".format(num_written_bboxes))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    coco_to_tfrecord(images_dir=args.images_dir,
                     annotations_file=args.annotations_file,
                     output=args.output,
                     num_shards=args.num_shards,
                     ltrb=args.ltrb,
                     ratio=args.ratio,
                     size_threshold=args.size_threshold,
                     skip_empty=args.skip_empty,
                     seed=args.seed)
