#!/bin/bash

WARMUP_EPOCHS=2.619685 # 300 iterations * 8 GPUs * 1 nodes * 128 batch size / 117266 non-empty images

mpirun -np 8 \
    -bind-to none \
    -map-by slot \
    --allow-run-as-root \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python ssd_main_async.py \
        --log-interval 20 \
        --batch-size 128 \
        --lr-warmup-epochs $WARMUP_EPOCHS \
        --pretrained-backbone /datasets/backbones/resnet34-333f7ec4.pickle \
        --data-layout NHWC \
        --precision amp
