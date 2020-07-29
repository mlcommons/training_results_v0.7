#!/usr/bin/env python

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

# NOTE: requires pytorch to read input file, so run something like:
# docker run --env PYTHONDONTWRITEBYTECODE=1 -v $(pwd):/scratch \
#        --ipc=host nvcr.io/nvidia/pytorch:19.09-py3            \
#        /scratch/torch-to-numpy.py /scratch/resnet34-333f7ec4.pth  /scratch/resnet34-333f7ec4.pickle

import torch
import pickle
from argparse import ArgumentParser

parser = ArgumentParser(description="read in pytorch .pth file and convert to pickled dictionary of numpy arrays")

parser.add_argument('input_file', type=str, help='input pytorch .pth file')
parser.add_argument('output_file', type=str, help='output pickled python file with dictionary of numpy arrays')
parser.add_argument('--verbose', action='store_true',
                    help='also spew the list of layers and their sizes')

args = parser.parse_args()

with open(args.input_file, 'rb') as pth_file:
    d = torch.load(pth_file)

dout = {}

for key, value in d.items():
    key_l = key.split('.')
    if key_l[0] == 'layer4' or key_l[0] == 'fc':
        continue
    new_val = value.data.numpy()
    assert(new_val.dtype == 'float32')

    stage = 0
    block = 0
    if key_l[0].startswith("layer"):
        # extract digits from the end of the string
        stage = int(list(filter(str.isdigit, key_l[0]))[0])
        block = int(key_l[1])
        key_l = key_l[2:]

    if key_l[0].startswith("downsample"):
        sub_block = 3
        key_l = key_l[2:]
    else:
        # extract digits from the end of the string
        sub_block = int(list(filter(str.isdigit, key_l[0]))[0])
        key_l = key_l[1:]

    sub_block = sub_block-1
    tensor_name = key_l[0]

    if tensor_name == "bias":
        tensor_name = "beta"
    elif tensor_name == "weight" and len(new_val.shape) == 1:
        tensor_name = "gamma"

    outname = "stage" + str(stage) + "_" if stage > 0 else ""
    outname = outname + ("conv" if len(new_val.shape) == 4 else "batchnorm")
    abs_subblock = 2*block + sub_block + (1 if ((stage > 1) and (block > 0)) else 0)
    outname = outname + str(abs_subblock) + "_" + tensor_name

    dout[outname] = new_val

    if args.verbose:
        print(outname, dout[outname].dtype, dout[outname].shape)

with open(args.output_file, 'wb') as out_file:
    pickle.dump(dout, out_file)
