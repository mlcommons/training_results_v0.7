#!/bin/bash

## DL params
OPTIMIZER="sgdwfastlars"
BATCHSIZE="208"
KVSTORE="horovod"
LR="7.4"
WARMUP_EPOCHS="2"
EVAL_OFFSET="2"
EVAL_PERIOD="4"
WD="0.0001"
LARSETA="0.001"
LABELSMOOTHING="0.1"
LRSCHED="pow2"
NUMEPOCHS="37"

NETWORK="resnet-v1b-normconv-fl"
export MXNET_CUDNN_SUPPLY_NORMCONV_CONSTANTS=1

DALI_PREFETCH_QUEUE="5"
DALI_NVJPEG_MEMPADDING="256"
DALI_CACHE_SIZE="0"
DALI_ROI_DECODE="1"  #needs to be set to 1 as default and proof perf uplift

## Environment variables for multi node runs
## TODO: These are settings for large scale runs that
## may need to be adjusted for single node.
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=20
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25

## System run parms
PGNNODES=1
PGSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=03:00:00

## System config params
PGNGPU=8
PGSOCKETCORES=24
PGHT=2         # HT is on is 2, HT off is 1
PGIBDEVICES=''
