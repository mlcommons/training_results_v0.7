source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
## DL params --64k
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="35"
export KVSTORE="horovod"
export LR="24.699"
export WARMUP_EPOCHS="31"
export EVAL_OFFSET="0" #Targeting epoch 89, 85, 81 ...
export EVAL_PERIOD="4"
export WD="0.0001"
export MOM="0.951807"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="90" 

export NETWORK="resnet-v1b-fl"
export MXNET_CONV_DGRAD_AS_GEMM=1

export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="12288"

# Default is no NCCL and BWD overlap
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
export MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1

## System run parms
export DGXNNODES=230
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:30:00
