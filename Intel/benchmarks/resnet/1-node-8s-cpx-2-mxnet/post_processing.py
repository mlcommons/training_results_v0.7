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
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Post processing for MLPerf MXNet logging')
    parser.add_argument("--in-file", type=str, required=True,
                        help="input logging file to be processed")
    parser.add_argument("--out-file", type=str, required=True,
                        help="output logging file")
    args = parser.parse_args()
    return args

args = parse_args()
f_read = open(args.in_file, "r")
f_write = open(args.out_file, "w")

for line in f_read.readlines():
    # replace single quote with double quote
    if "'" in line:
        line = line.replace("'", '"')
    # replace "training_samples" with "train_samples"
    if "training_samples" in line:
        line = line.replace("training_samples", "train_samples")
    # replace "evaluation_samples" with "eval_samples"
    if "evaluation_samples" in line:
        line = line.replace("evaluation_samples", "eval_samples")
    # remove the square buckets around eval_accuracy
    if "eval_accuracy" in line:
        pos1 = line.find("[")
        pos2 = line.find("]")
        line = line[:pos1-1] + line[pos1+1:pos2] + line[pos2+2:]

    f_write.write(line)
    

f_read.close()
f_write.close()