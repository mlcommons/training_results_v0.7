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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:20.06-py3
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*

# Update container's pycocotools to optimized version
RUN pip uninstall -y pycocotools

ENV COCOAPI_VERSION=2.0+nv0.4.0
RUN export COCOAPI_TAG=$(echo ${COCOAPI_VERSION} | sed 's/^.*+n//') \
 && pip install --no-cache-dir pybind11                             \
 && pip install --no-cache-dir git+https://github.com/NVIDIA/cocoapi.git@${COCOAPI_TAG}#subdirectory=PythonAPI

WORKDIR /workspace/ssd

# Install python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir cython \
 && pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip install --no-cache-dir -r requirements.txt

# Copy SSD code
COPY . .

# Compile Horovod MPI test
RUN cd /workspace/ssd/tests && mpicxx --std=c++11 horovod_mpi_test.cpp -o horovod_mpi_test && cd /workspace/ssd

ENV HOROVOD_CYCLE_TIME=0.1                      \
    HOROVOD_BATCH_D2D_MEMCOPIES=1               \
    HOROVOD_NUM_NCCL_STREAMS=1                  \
    MXNET_CUDNN_AUTOTUNE_DEFAULT=0              \
    MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999 \
    MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999 \
    MXNET_GPU_WORKER_NTHREADS=1                 \
    MXNET_CPU_PRIORITY_NTHREADS=1               \
    MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1      \
    OMP_NUM_THREADS=1                           \
    OMPI_MCA_btl=^openib                        \
    OPENCV_FOR_THREADS_NUM=1

#    HOROVOD_STALL_CHECK_DISABLE=1
