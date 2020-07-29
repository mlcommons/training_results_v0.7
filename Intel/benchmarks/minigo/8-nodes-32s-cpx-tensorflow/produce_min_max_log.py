# Copyright 2019 Google LLC
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

#!/usr/bin/env python
# encoding: utf-8

import time
import os
import logging

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile

from absl import app, flags

import preprocessing
import dual_net


flags.DEFINE_string('input_graph', None, 'The path of input graph.')
flags.DEFINE_string('data_location', None, 'The path of input data.')
flags.DEFINE_integer('num_steps', 20, 'Number of eval steps.')
flags.DEFINE_integer('batch_size', 20, 'eval batch size.')
flags.DEFINE_boolean('random_rotation', True, 'Do random rotation if true.')

FLAGS = flags.FLAGS

def run_graph(graph, tf_records):

  data_graph = tf.Graph()
  with data_graph.as_default():
    features, labels = preprocessing.get_input_tensors(
              FLAGS.batch_size,
              FLAGS.input_layout,
              tf_records,
              shuffle_buffer_size=100000000,
              random_rotation=FLAGS.random_rotation,
              make_one_shot=True,
              use_bf16=False)

  infer_graph = tf.Graph()
  with infer_graph.as_default():
    tf.import_graph_def(graph, name='')

  input_tensor = dual_net.get_input_tensor(infer_graph)
  output_tensor = dual_net.get_output_tensor(infer_graph)

  config = tf.compat.v1.ConfigProto()
  data_sess = tf.compat.v1.Session(graph=data_graph, config=config)
  infer_sess = tf.compat.v1.Session(graph=infer_graph, config=config)

  elapsed = 0
  for it in range(FLAGS.num_steps):
    features_np = data_sess.run(features)
    start_time = time.time()
    infer_sess.run(output_tensor, feed_dict={input_tensor: features_np})
    elapsed += time.time() - start_time

def read_graph(input_graph):
  if not gfile.Exists(input_graph):
    logging.info("Input graph file '" + input_graph + "' does not exist!")
    exit(-1)

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(input_graph, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

  return input_graph_def


def main(unused_argv):
  """Run the reinforcement learning loop."""
  graph = read_graph(FLAGS.input_graph)
  tf_records = sorted(tf.io.gfile.glob(FLAGS.data_location), reverse=True)[:1]
  run_graph(graph, tf_records)

if __name__ == "__main__":
    app.run(main)
