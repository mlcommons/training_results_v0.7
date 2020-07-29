source $(dirname ${BASH_SOURCE[0]})/config_NF5488_common.sh

export DALI_THREADS=8

## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="408"
export KVSTORE="horovod"
export LR="10.5"
export WARMUP_EPOCHS="2"
export EVAL_OFFSET="2" # Targeting epoch no. 35
export EVAL_PERIOD="4"
export WD="0.00005"
export MOM="0.9"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="37"

export NETWORK="resnet-v1b-stats-fl"

export DALI_PREFETCH_QUEUE="5"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="0"
export DALI_ROI_DECODE="1"  #needs to be set to 1 as default and proof perf uplift

export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=20
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25
export MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1
export NCCL_MAX_RINGS=8

## System run parms
export NFNNODES=1
export NFSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:00:00
