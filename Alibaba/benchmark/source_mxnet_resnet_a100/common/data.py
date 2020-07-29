# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import tempfile
import mxnet as mx
import random
import argparse
from mxnet.io import DataBatch, DataIter
import numpy as np
import warnings
import horovod.mxnet as hvd
from common import dali

def build_input_pipeline(args, kv=None):
    return lambda args, kv: get_rec_iter(args, kv)

class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype, layout='NHWC'):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        self.layout = layout
        label = np.random.randint(0, num_classes, [self.batch_size,])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=np.float16, ctx=mx.Context('gpu'))
        self.label = mx.nd.array(label, dtype=self.dtype, ctx=mx.Context('gpu'))
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, np.float16, self.layout)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0

def get_rec_iter(args, kv=None):
    (rank, num_workers) = dali._get_rank_and_worker_count(args, kv)
    examples = args.num_examples // num_workers
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if args.input_layout == 'NHWC':
        image_shape = image_shape[1:] + (image_shape[0],)
    data_shape = (args.batch_size,) + image_shape
    train = SyntheticDataIter(args.num_classes, data_shape,
            examples / args.batch_size, np.float32, args.input_layout)
    return (train, None)
