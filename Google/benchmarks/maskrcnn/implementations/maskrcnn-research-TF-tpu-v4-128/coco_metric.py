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
"""COCO-style evaluation metrics.

Implements the interface of COCO API and metric_fn in tf.TPUEstimator.

COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import copy
import tempfile
import numpy as np

# copybara:strip_begin
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# copybara:strip_end
import pycocotools.mask as maskUtils
# copybara:strip_begin
from pycocotools.python import coco
from pycocotools.python import cocoeval
# copybara:strip_end
# copybara:insert import coco

import tensorflow.compat.v1 as tf


# copybara:insert COCO = coco.COCO
# copybara:insert COCOeval = coco.COCOeval


class MaskCOCO(COCO):
  """COCO object for mask evaluation.
  """

  def loadRes(self, detection_results, mask_results):
    """Load result file and return a result api object.

    Args:
      detection_results: a numpy array of detection results of shape:
        [num_images * detection_per_image, 7]. The format is:
        [image_id, x, y, width, height, score, class].
      mask_results: a list of RLE encoded binary instance masks. Length is
        num_images * detections_per_image.

    Returns:
      res: result MaskCOCO api object
    """
    res = MaskCOCO()
    res.dataset['images'] = [img for img in self.dataset['images']]
    print('Loading and preparing results...')
    predictions = self.load_predictions(detection_results, mask_results)
    assert isinstance(predictions, list), 'results in not an array of objects'

    image_ids = [pred['image_id'] for pred in predictions]
    assert set(image_ids) == (set(image_ids) & set(self.getImgIds())), \
           'Results do not correspond to current coco set'

    if ('bbox' in predictions[0] and predictions[0]['bbox']):
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for idx, pred in enumerate(predictions):
        bb = pred['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
        if 'segmentation' not in pred:
          pred['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        pred['area'] = bb[2]*bb[3]
        pred['id'] = idx+1
        pred['iscrowd'] = 0
    elif 'segmentation' in predictions[0]:
      res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
      for idx, pred in enumerate(predictions):
        # now only support compressed RLE format as segmentation results
        pred['area'] = maskUtils.area(pred['segmentation'])
        if 'bbox' not in pred:
          pred['bbox'] = maskUtils.toBbox(pred['segmentation'])
        pred['id'] = idx+1
        pred['iscrowd'] = 0

    res.dataset['annotations'] = predictions
    res.createIndex()
    return res

  def load_predictions(self, detection_results, mask_results):
    """Create prediction dictionary list from detection and mask results.

    Args:
      detection_results: a numpy array of detection results of shape:
        [num_images * detection_per_image, 7].
      mask_results: a list of RLE encoded binary instance masks. Length is
        num_images * detections_per_image.

    Returns:
      annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert isinstance(detection_results, np.ndarray)
    print(detection_results.shape[0])
    print(len(mask_results))
    assert detection_results.shape[1] == 7
    assert detection_results.shape[0] == len(mask_results)
    num_detections = detection_results.shape[0]
    predictions = []

    for i in range(num_detections):
      if i % 1000000 == 0:
        print('{}/{}'.format(i, num_detections))
      predictions += [{
          'image_id': int(detection_results[i, 0]),
          'bbox': detection_results[i, 1:5].tolist(),
          'score': detection_results[i, 5],
          'category_id': int(detection_results[i, 6]),
          'segmentation': mask_results[i],
          }]
    return predictions


class EvaluationMetric(object):
  """COCO evaluation metric class."""

  def __init__(self, filename, use_cpp_extension=False):
    """Constructs COCO evaluation class.

    The class provides the interface to metrics_fn in TPUEstimator. The
    _update_op() takes detections from each image and push them to
    self.detections. The _evaluate() loads a JSON file in COCO annotation format
    as the groundtruths and runs COCO evaluation.

    Args:
      filename: Ground truth JSON file name. If filename is None, use
        groundtruth data passed from the dataloader for evaluation.
      use_cpp_extension: use cocoeval C++ library.
    """
    if filename:
      if filename.startswith('gs://'):
        _, local_val_json = tempfile.mkstemp(suffix='.json')
        tf.gfile.Remove(local_val_json)

        tf.gfile.Copy(filename, local_val_json)
        atexit.register(tf.gfile.Remove, local_val_json)
      else:
        local_val_json = filename
      self.use_cpp_extension = use_cpp_extension
      if self.use_cpp_extension:
        self.coco_gt = coco.COCO(local_val_json, use_mask=True)
      else:
        self.coco_gt = MaskCOCO(local_val_json)
    self.filename = filename
    self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                         'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
    mask_metric_names = ['mask_' + x for x in self.metric_names]
    self.metric_names.extend(mask_metric_names)

    self._reset()

  def _reset(self):
    """Reset COCO API object."""
    if self.filename is None:
      self.coco_gt = MaskCOCO()
    self.detections = []
    self.masks = []

  def evaluate(self):
    """Evaluates with detections from all images with COCO API.

    Returns:
      num_eval_samples: number of samples being evaludated.
      coco_metric: float numpy array with shape [24] representing the
        coco-style evaluation metrics (box and mask).
    """
    detections = np.array(self.detections)
    concat_masks = [x for img_masks in self.masks for x in img_masks]
    image_ids = list(set(detections[:, 0]))

    if self.use_cpp_extension:
      coco_dt = self.coco_gt.LoadResMask(detections, concat_masks)
      # copybara:strip_begin
      coco_eval = cocoeval.COCOeval(self.coco_gt, coco_dt, iou_type='bbox')
      # copybara:strip_end
      # pylint: disable=line-too-long
      # copybara:insert coco_eval = COCOeval(self.coco_gt, coco_dt, iou_type='bbox')
      # pylint: enable=line-too-long
      coco_eval.Evaluate()
      coco_eval.Accumulate()
      coco_eval.Summarize()
      coco_metrics = coco_eval.GetStats()

      # Create another object for instance segmentation metric evaluation.
      # copybara:strip_begin
      mcoco_eval = cocoeval.COCOeval(self.coco_gt, coco_dt, iou_type='segm')
      # copybara:strip_end
      # pylint: disable=line-too-long
      # copybara:insert mcoco_eval = COCOeval(self.coco_gt, coco_dt, iou_type='segm')
      # pylint: enable=line-too-long
      mcoco_eval.Evaluate()
      mcoco_eval.Accumulate()
      mcoco_eval.Summarize()
      mask_coco_metrics = mcoco_eval.GetStats()
    else:
      coco_dt = self.coco_gt.loadRes(detections, concat_masks)
      coco_eval = COCOeval(self.coco_gt, coco_dt, iouType='bbox')
      coco_eval.params.imgIds = image_ids
      coco_eval.evaluate()
      coco_eval.accumulate()
      coco_eval.summarize()
      coco_metrics = coco_eval.stats

      # Create another object for instance segmentation metric evaluation.
      mcoco_eval = COCOeval(self.coco_gt, coco_dt, iouType='segm')
      mcoco_eval.params.imgIds = image_ids
      mcoco_eval.evaluate()
      mcoco_eval.accumulate()
      mcoco_eval.summarize()
      mask_coco_metrics = mcoco_eval.stats

    combined_metrics = np.hstack((coco_metrics, mask_coco_metrics))
    eval_results = {}
    for i, name in enumerate(self.metric_names):
      eval_results[name] = combined_metrics[i].astype(np.float32)
    # clean self.detections after evaluation is done.
    # this makes sure the next evaluation will start with an empty list of
    # self.detections.
    num_eval_samples = len(self.masks)
    self._reset()
    return num_eval_samples, eval_results

  def update(self, detections, segmentations):
    """Updates detections and segmentations."""
    self.detections.extend(detections)
    self.masks.extend(segmentations)
