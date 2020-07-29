import time
import logging
from functools import lru_cache

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@lru_cache(maxsize=None)
def get_coco_gt(annotation_file):
    cocoGt = COCO(annotation_file=annotation_file, use_ext=True)
    ssdid_to_cocoid = {v: k for v, k in enumerate(cocoGt.getCatIds())}
    return cocoGt, ssdid_to_cocoid


# TODO(ahmadki): remove time measurements
def coco_map_score(results, annotation_file, num_threads=1):
    time_ticks = [time.time()]
    time_messages = []
    cocoGt, ssdid_to_cocoid = get_coco_gt(annotation_file=annotation_file)
    time_ticks.append(time.time())
    time_messages.append(['get_coco_gt time'])

    pred_bboxes = results[:, 0:4,].astype(dtype='float32')
    pred_scores = results[:, 4:5,].astype(dtype='float32')
    pred_labels = results[:, 5:6,].astype(dtype='int32')
    img_ids = results[:, 6:7,].astype(dtype='int32')
    time_ticks.append(time.time())
    time_messages.append('numpy slice time')

    # prepare prediction results
    pred_labels = np.vectorize(ssdid_to_cocoid.get)(pred_labels)
    pred_results = np.concatenate((img_ids, pred_bboxes, pred_scores, pred_labels), axis=1)
    time_ticks.append(time.time())
    time_messages.append('numpy to cocoapi format time')

    cocoDt = cocoGt.loadRes(pred_results, use_ext=True)
    time_ticks.append(time.time())
    time_messages.append('cocoGt.loadRes')

    E = COCOeval(cocoGt, cocoDt, iouType='bbox', num_threads=num_threads, use_ext=True)
    time_ticks.append(time.time())
    time_messages.append('COCOeval')

    E.evaluate()
    time_ticks.append(time.time())
    time_messages.append('E.evaluate')

    E.accumulate()
    time_ticks.append(time.time())
    time_messages.append('E.accumulate')

    E.summarize()
    time_ticks.append(time.time())
    time_messages.append('E.summarize')

    time_ticks = np.array(time_ticks)
    elpased_time = time_ticks[1:]-time_ticks[:-1]
    for msg, t in zip(time_messages, elpased_time):
        logging.info(f'{msg}: {t} [sec]')

    validation_map = E.stats[0]*100
    logging.info(f'validation_map: {validation_map}')

    return validation_map
