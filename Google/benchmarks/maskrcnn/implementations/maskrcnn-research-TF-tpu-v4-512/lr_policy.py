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

import tensorflow.compat.v1 as tf
from REDACTED.mlp_log import mlp_log


def learning_rate_schedule(peak_learning_rate, lr_warmup_init,
                           lr_warmup_step, first_lr_drop_step,
                           second_lr_drop_step, global_step):
  """Handles linear scaling rule, gradual warmup, and LR decay."""
  # lr_warmup_init is the starting learning rate; the learning rate is linearly
  # scaled up to the full learning rate after `lr_warmup_step` before decaying.
  mlp_log.mlperf_print(key='opt_learning_rate_decay_factor', value=0.1)
  mlp_log.mlperf_print('opt_learning_rate_decay_steps',
                       (first_lr_drop_step, second_lr_drop_step))
  linear_warmup = (lr_warmup_init +
                   (tf.cast(global_step, dtype=tf.float32) / lr_warmup_step *
                    (peak_learning_rate - lr_warmup_init)))
  learning_rate = tf.where(global_step < lr_warmup_step,
                           linear_warmup, peak_learning_rate)
  lr_schedule = [[1.0, lr_warmup_step],
                 [0.1, first_lr_drop_step],
                 [0.01, second_lr_drop_step]]
  for mult, start_global_step in lr_schedule:
    learning_rate = tf.where(global_step < start_global_step, learning_rate,
                             peak_learning_rate * mult)
  return learning_rate
