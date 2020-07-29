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
"""Use multiprocess to perform COCO metric evaluation.
"""
# copybara:insert import multiprocessing
from REDACTED.tensorflow_models.mlperf.models.rough.mask_rcnn import mask_rcnn_params
from REDACTED.tensorflow_models.mlperf.models.rough.mask_rcnn import segm_utils
# copybara:strip_begin
from REDACTED.REDACTED.multiprocessing import REDACTEDprocess
# copybara:strip_end


# copybara:strip_begin
def REDACTED_post_processing():
  """REDACTED batch-processes the predictions."""
  q_in, q_out = REDACTEDprocess.get_user_data()
  post_processing(q_in, q_out)
# copybara:strip_end


def post_processing(q_in, q_out):
  """Batch-processes the predictions."""
  boxes, masks, image_info = q_in.get()
  while boxes is not None:
    detections = []
    segmentations = []
    for i, box in enumerate(boxes):
      # Slice out the padding data where score is zero
      num = max(1, sum(box[:, 5] > 0))
      box = box[:num, :]
      segms = segm_utils.segm_results(
          masks[i], box[:, 1:5], int(image_info[i][3]), int(image_info[i][4]))
      detections.extend(box)
      segmentations.append(segms)
    q_out.put((detections, segmentations))
    boxes, masks, image_info = q_in.get()
  # signal the parent process that we have completed all work.
  q_out.put((None, None))


def update_eval_metric(q_out, eval_metric, exited_process):
  detections, segmentations = q_out.get()
  if detections is None and segmentations is None:
    exited_process += 1
  else:
    eval_metric.update(detections, segmentations)
  return exited_process


def eval_multiprocessing(predictions,
                         eval_metric,
                         eval_worker_count,
                         queue_size=mask_rcnn_params.QUEUE_SIZE):
  """Enables multiprocessing to update eval metrics."""
  # copybara:strip_begin
  q_in, q_out = REDACTEDprocess.get_user_data()
  processes = [
      REDACTEDprocess.Process(target=REDACTED_post_processing)
      for _ in range(eval_worker_count)
  ]
  # copybara:strip_end_and_replace_begin
  # q_in = multiprocessing.Queue(maxsize=queue_size)
  # q_out = multiprocessing.Queue(maxsize=queue_size)
  # processes = [
  #     multiprocessing.Process(target=post_processing, args=(q_in, q_out))
  #     for _ in range(eval_worker_count)
  # ]
  # copybara:replace_end
  for p in processes:
    p.start()

  # TODO(b/129410706): investigate whether threading improves speed.
  # Every predictor.next() gets a batch of prediction (a dictionary).
  exited_process = 0
  samples = len(predictions['detections']) // eval_worker_count
  for i in range(eval_worker_count):
    while q_in.full() or q_out.qsize() > queue_size // 4:
      exited_process = update_eval_metric(q_out, eval_metric, exited_process)

    q_in.put((predictions['detections'][i * samples:(i + 1) * samples],
              predictions['mask_outputs'][i * samples:(i + 1) * samples],
              predictions['image_info'][i * samples:(i + 1) * samples]))

  # Adds empty items to signal the children to quit.
  for _ in processes:
    q_in.put((None, None, None))

  # Cleans up q_out and waits for all the processes to finish work.
  while not q_out.empty() or exited_process < eval_worker_count:
    exited_process = update_eval_metric(q_out, eval_metric, exited_process)

  for p in processes:
    # actively terminate all processes (to work around the multiprocessing
    # deadlock issue in Cloud)
    # copybara:insert p.terminate()
    p.join()
