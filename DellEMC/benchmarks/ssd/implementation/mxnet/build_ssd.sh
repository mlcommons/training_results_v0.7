#!/bin/bash

# GPU driver path
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH


# Update container's pycocotools to optimized version
pip uninstall -y pycocotools

export COCOAPI_VERSION=2.0+nv0.4.0
export COCOAPI_TAG=$(echo ${COCOAPI_VERSION} | sed 's/^.*+n//') 

pip install --no-cache-dir pybind11                             
pip install --no-cache-dir git+https://github.com/NVIDIA/cocoapi.git@${COCOAPI_TAG}#subdirectory=PythonAPI

cd /mnt/current

pip install --no-cache-dir cython 
pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip 
pip install --no-cache-dir -r requirements.txt

# Compile Horovod MPI test
cd tests 
mpicxx --std=c++11 horovod_mpi_test.cpp -o horovod_mpi_test 


