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
from REDACTED.tensorflow.contrib import training as contrib_training

BOX_EVAL_TARGET = 0.377
MASK_EVAL_TARGET = 0.339

EVAL_WORKER_COUNT = 20
# make sure size of worker queue is larger than EVAL_WORKER_COUNT + eval_steps
QUEUE_SIZE = 48

IS_PADDED = 'is_padded'


def default_hparams():
  return contrib_training.HParams(
      # input preprocessing parameters
      image_size=(832, 1344),
      short_side_image_size=800,
      long_side_max_image_size=1333,
      input_rand_hflip=True,
      gt_mask_size=112,
      shuffle_buffer_size=4096,
      # dataset specific parameters
      num_classes=91,
      skip_crowd_during_training=True,
      use_category=True,
      # Region Proposal Network
      rpn_positive_overlap=0.7,
      rpn_negative_overlap=0.3,
      rpn_batch_size_per_im=256,
      rpn_fg_fraction=0.5,
      rpn_pre_nms_topn=2000,
      rpn_post_nms_topn=1000,
      rpn_nms_threshold=0.7,
      rpn_min_size=0.,
      # Proposal layer.
      batch_size_per_im=512,
      fg_fraction=0.25,
      fg_thresh=0.5,
      bg_thresh_hi=0.5,
      bg_thresh_lo=0.,
      # Faster-RCNN heads.
      fast_rcnn_mlp_head_dim=1024,
      bbox_reg_weights=(10., 10., 5., 5.),
      # Mask-RCNN heads.
      mrcnn_resolution=28,
      # evaluation
      test_detections_per_image=100,
      test_nms=0.5,
      test_rpn_pre_nms_topn=1000,
      test_rpn_post_nms_topn=1000,
      test_rpn_nms_thresh=0.7,
      # model architecture
      min_level=2,
      max_level=6,
      num_scales=1,
      aspect_ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
      anchor_scale=8.0,
      resnet_depth=50,
      # is batchnorm training mode
      is_training_bn=False,
      # localization loss
      delta=0.1,
      rpn_box_loss_weight=1.0,
      fast_rcnn_box_loss_weight=1.0,
      mrcnn_weight_loss_mask=1.0,
      # optimization, default for batch size=128
      momentum=0.9,
      learning_rate=0.16,
      lr_warmup_init=0.,
      lr_warmup_step=1000,
      first_lr_drop_step=7500,
      second_lr_drop_step=10000,
      # training configuration
      num_examples_per_epoch=118287,
      eval_samples=5000,
      # enable bfloat
      use_bfloat16=True,
      # TPU performance optimization
      transpose_input=True,
      eval_use_tpu_estimator=False,
      train_use_tpu_estimator=False,
      # conv0 optimization
      conv0_kernel_size=7,
      conv0_space_to_depth_block_size=2,
      # enable host call.
      use_host_call=False,
      all_in_one_session=True,
      train_and_eval_save_checkpoint=True,
      eval_worker_count=EVAL_WORKER_COUNT,
      hosts_per_dataset_shard=1,
      # for spatial partition.
      use_spmd=True,
  )
