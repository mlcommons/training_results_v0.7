#!/bin/bash

cd ../implementation_open/mxnet

pip install cmake
conda install nasm
conda install opencv
conda install openblas

#DALI
pip install --no-cache-dir --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali

#mpi4py
pip install --no-cache-dir mpi4py
make -j 24

#Build MXNet 
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
cd python
pip install -e .

#HOROVOD
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod==0.19.0

#Logger
pip install --no-cache-dir "git+https://github.com/mlperf/logging.git@0.7.0-rc3-closed-division"
