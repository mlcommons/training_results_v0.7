source $(dirname ${BASH_SOURCE[0]})/config_DSS8440_common.sh
## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="208"
export KVSTORE="horovod"
export LR="7.4"
export WARMUP_EPOCHS="2"
export EVAL_OFFSET="2"
export EVAL_PERIOD="4"
export WD="1e-04"
export MOM="0.9"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="37" 

#export NETWORK="resnet-v1b-dbar-fl"
#export NETWORK="resnet-v1b-normconv-fl"
export NETWORK="resnet-v1b-dbar-fl"
export MXNET_CUDNN_SUPPLY_NORMCONV_CONSTANTS=1

export DALI_PREFETCH_QUEUE="5"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="0"
export DALI_ROI_DECODE="1"  #needs to be set to 1 as default and proof perf uplift

## Environment variables for multi node runs
## TODO: These are settings for large scale runs that
## may need to be adjusted for single node.
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
export NNODES=1
export SYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:30:00

## System config params
export NGPU=8
export SOCKETCORES=20
export HT=2         # HT is on is 2, HT off is 1

export TOBIND=0
export NCCL_SOCKET_IFNAME=
