# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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


# NOTE: This implementation of DLRM uses the HugeCTR framework.  The benchmark
# leverages the NGC PyTorch image solely for the GPU-enabled build environment
# it provides, not for the PyTorch framework itself (HugeCTR as a framework is
# independent of PyTorch).  Other NGC DL Frameworks images such as
# nvcr.io/nvidia/tensorflow:20.06-tf1-py3 or nvcr.io/nvidia/mxnet:20.06-py3
# could have been used similarly as a base for this benchmark.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}

# Install HugeCTR framework
WORKDIR /opt
RUN git clone --recursive -b v2.1_a100_update https://github.com/NVIDIA/hugectr.git \
 && cd hugectr \
 && mkdir -p build \
 && cd build \
 && cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" .. \
 && make -j$(nproc) \
 && cp bin/* /usr/local/bin \
 && cd .. \
 && git submodule deinit json \
 && rm -rf build .git/modules/json

# Install Python dependencies
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip install --no-cache-dir numpy pandas sklearn ortools

# Install DLRM benchmark code
WORKDIR /workspace/dlrm
COPY . .
