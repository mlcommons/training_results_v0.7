#!/bin/bash

sandbox="sandbox-mxnet-ngc20.06"
#singularity pull mxnet_20.06-py3.sif docker://nvcr.io/nvidia/mxnet:20.06-py3
singularity build --sandbox $sandbox mxnet_20.06-py3.sif
singularity exec -w $sandbox mkdir -p /mnt/current /mnt/driver /data /pretrained/mxnet
#singularity exec -w -B /cm/local/apps/cuda/libs/current:/mnt/driver -B $PWD:/mnt/current $sandbox bash /mnt/current/build_ssd.sh
