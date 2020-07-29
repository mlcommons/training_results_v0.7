# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:20.06-tf1-py3
FROM ${FROM_IMAGE_NAME}

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        cmake \
        python2.7 \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*

# Copy MiniGo code
WORKDIR /opt/reinforcement
COPY . .

ENV BOARD_SIZE 19
ENV MINIGO_BAZEL_CACHE_DIR /opt/reinforcement/minigo-bazel-cache

# Copy TF dependency
RUN mkdir minigo/cc/tensorflow/lib \
 && cp /usr/local/lib/python3.6/dist-packages/tensorflow_core/libtensorflow_framework.so.1 minigo/cc/tensorflow/lib \
 && cp /usr/local/lib/tensorflow/libtensorflow_cc.so.1 minigo/cc/tensorflow/lib \
 && cp -r /usr/local/lib/python3.6/dist-packages/tensorflow_core/include minigo/cc/tensorflow/include

# Install Python dependencies
WORKDIR /opt/reinforcement/minigo
RUN pip3 install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip3 install --no-cache-dir -r requirements.txt

# Install pybind11 to enable C++-python interface
RUN pip3 install --no-cache-dir pytest \
 && git clone --branch v2.4.3 --depth 1 https://github.com/pybind/pybind11 /usr/local/src/pybind11 \
 && cd /usr/local/src/pybind11 \
 && cmake . \
 && make install -j$(nproc) \
 && pip3 install --no-cache-dir .

# Build Minigo
RUN mkdir -p "${MINIGO_BAZEL_CACHE_DIR}" \
 && bazel --output_user_root="${MINIGO_BAZEL_CACHE_DIR}" build -c opt \
      --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
      --copt=-O3 \
      --define=board_size="${BOARD_SIZE}" \
      --define=tf=1 \
      cc:minigo_python.so

ENV PYTHONPATH "${PYTHONPATH}:/opt/reinforcement/minigo/bazel-bin/cc"
RUN echo '/usr/local/lib/python3.6/dist-packages/tensorflow_core' > /etc/ld.so.conf.d/tensorflow.conf && ldconfig

# back to where run* files are
WORKDIR /opt/reinforcement
