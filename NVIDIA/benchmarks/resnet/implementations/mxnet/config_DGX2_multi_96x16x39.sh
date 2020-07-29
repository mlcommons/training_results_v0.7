source $(dirname ${BASH_SOURCE[0]})/config_DGX2_common.sh
## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="39"
export KVSTORE="horovod"
export LR="26"
export WARMUP_EPOCHS="33"
export EVAL_OFFSET="1" # Targeting epoch no 90,86,82
export EVAL_PERIOD="4"
export WD="2.5e-05"
export MOM="0.94"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="90"

export NETWORK="resnet-v1b-dbar-fl"
export MXNET_CUDNN_SUPPLY_NORMCONV_CONSTANTS=1
export MXNET_CONV_DGRAD_AS_GEMM=1

export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE=6144

## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999
export MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1

## System run parms
export DGXNNODES=96
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:20:00

## OMPI
export OMPI_MCA_btl="^openib"

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXHT=2 	# HT is on is 2, HT off is 1
