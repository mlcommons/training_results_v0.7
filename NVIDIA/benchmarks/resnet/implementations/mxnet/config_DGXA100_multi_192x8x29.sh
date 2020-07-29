source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
## DL params -- GBS 43k
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="29"
export KVSTORE="horovod"
export LR="23.5"
export WARMUP_EPOCHS="22"
export EVAL_OFFSET="1"
export EVAL_PERIOD="4"
export WD="0.000025"
export MOM="0.94"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="70"

export NETWORK="resnet-v1b-fl"

export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="12288"

export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
export MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1

## System run parms
export DGXNNODES=192
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:15:00
