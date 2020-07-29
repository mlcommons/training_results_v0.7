# Lint as: python3
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests BERT pretraining."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import REDACTED
from absl import flags
from absl import logging
from absl.testing import flagsaver

import tensorflow as tf

from REDACTED.tf2_bert.bert import run_pretraining

_BERT_CONFIG_FILE = ('/REDACTED/od-d/home/jacobdevlin/public/bert/pretrained_models/'
                     'uncased_L-24_H-1024_A-16/bert_config.json')
_INPUT_FILES = ('/REDACTED/is-d/home/tpu-perf-team/sgpyc/bert/'
                'seq_512_mpps_76_tfrecords3/part-*')


FLAGS = flags.FLAGS


def set_up_flags():
  """Basic test flags."""
  FLAGS.max_seq_length = 512
  FLAGS.max_predictions_per_seq = 76
  FLAGS.train_batch_size = 2
  FLAGS.eval_batch_size = 2
  FLAGS.num_eval_samples = 1
  FLAGS.learning_rate = 0.000625
  FLAGS.warmup_steps = 1
  FLAGS.num_steps_per_epoch = 2
  FLAGS.num_train_epochs = 1
  FLAGS.steps_between_eval = 2
  FLAGS.steps_per_loop = 2
  FLAGS.device_warmup = True
  FLAGS.do_eval = True
  FLAGS.bert_config_file = _BERT_CONFIG_FILE
  FLAGS.optimizer_type = 'lamb'
  FLAGS.input_files = _INPUT_FILES
  FLAGS.dtype = 'bf16'
  FLAGS.distribution_strategy = 'tpu'
  FLAGS.tpu = ''


class EndToEndTests(tf.test.TestCase):
  """Unit tests for BERT pretratining."""

  def setUp(self):
    super(EndToEndTests, self).setUp()
    self.policy = \
        tf.compat.v2.keras.mixed_precision.experimental.global_policy()

  def tearDown(self):
    super(EndToEndTests, self).tearDown()
    tf.io.gfile.rmtree(self.get_temp_dir())
    tf.compat.v2.keras.mixed_precision.experimental.set_policy(self.policy)

  @flagsaver.flagsaver
  def test_end_to_end(self):
    set_up_flags()
    model_dir = os.path.join(self.get_temp_dir(), 'pretrain_test')
    FLAGS.model_dir = model_dir
    run_pretraining.main(None)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
