#!/bin/bash

DOWNLOAD_LINK='https://download.pytorch.org/models/resnet34-333f7ec4.pth'
FILENAME='resnet34-333f7ec4.pth'

wget -c $DOWNLOAD_LINK
docker run -it --rm \
    --gpus=all \
    --ipc=host \
    --env PYTHONDONTWRITEBYTECODE=1 \
    -v $(pwd):/scratch \
    nvcr.io/nvidia/pytorch:20.03-py3 \
    python /scratch/torch_to_numpy.py /scratch/resnet34-333f7ec4.pth /scratch/resnet34-333f7ec4.pickle
