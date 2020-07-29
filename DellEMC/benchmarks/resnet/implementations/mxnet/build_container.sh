#!/bin/bash

sandbox="sandbox-mxnet-ngc20.06"
#singularity pull mxnet_20.06-py3.sif docker://nvcr.io/nvidia/mxnet:20.06-py3
singularity build --sandbox $sandbox mxnet_20.06-py3.sif
singularity exec -w $sandbox mkdir -p /mnt/current /mnt/driver /data
singularity exec -w -B $PWD:/mnt/current $sandbox pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip 
singularity exec -w -B $PWD:/mnt/current $sandbox pip install --no-cache-dir -r /mnt/current/requirements.txt
