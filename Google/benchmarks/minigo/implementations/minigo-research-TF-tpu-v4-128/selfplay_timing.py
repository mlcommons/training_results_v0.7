# Lint as: python3
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Selfplay timing helper."""

import itertools
import logging
import os
import time

from absl import app
from absl import flags
import tensorflow as tf

from REDACTED.minigo import dual_net

flags.DEFINE_string('raw_model_path', None, 'Path to model to freeze')
flags.DEFINE_string('model_dir', None, 'Path to export frozen model')
flags.DEFINE_string('selfplay_dir', None, 'Path to export frozen model')
flags.DEFINE_string('abort_file_path', None, 'Path to export frozen model')
flags.DEFINE_integer('num_games', 8192, 'Path to export frozen model')

FLAGS = flags.FLAGS


def place_model(model_name):
  """freeze a new model and put in place for selfplay to grab."""
  FLAGS.work_dir = FLAGS.raw_model_path
  dual_net.freeze_graph_tpu(os.path.join(FLAGS.model_dir, model_name))


def time_selfplay(selfplay_path):
  """Log timestamps of selfplay progress."""
  pattern = os.path.join(selfplay_path, '*', '*', '*.tfrecord.zz')
  start_time = time.time()
  found_model = False
  for i in itertools.count():
    try:
      paths = tf.io.gfile.glob(pattern)
    except tf.errors.OpError:
      paths = []
    if not found_model and paths:
      found_model_time = time.time()
      logging.info(
          'Takes %d secs from when the model is placed to when the first game is generated',
          found_model_time - start_time)
      found_model = True
    if len(paths) >= FLAGS.num_games:
      logging.info('Done playing. Generated %d games in %d secs', len(paths),
                   time.time() - found_model_time)
      break
    if i % 10 == 0:
      logging.info('Waiting for %d games in %s (found %d)', FLAGS.num_games,
                   selfplay_path, len(paths))
    time.sleep(0.5)


def main(unused_argv):
  for i in range(3):
    model_name = '%06d' % i
    selfplay_dir_path = os.path.join(FLAGS.selfplay_dir, '%06d' % i)
    place_model(model_name)
    time_selfplay(selfplay_dir_path)
  with tf.io.gfile.GFile(FLAGS.abort_file_path, 'w') as f:
    f.write('abort')


if __name__ == '__main__':
  app.run(main)
