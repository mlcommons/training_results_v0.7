source $(dirname ${BASH_SOURCE[0]})/config_DGX1_common.sh
## DL params
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="208"
export KVSTORE="horovod"
export LR="7.4"
export WARMUP_EPOCHS="2"
export EVAL_OFFSET="0"
export EVAL_PERIOD="4"
export WD="1e-04"
export MOM="0.9"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS="37" 

export NETWORK="resnet-v1b-dbar-fl"
export MXNET_CUDNN_SUPPLY_NORMCONV_CONSTANTS=1
export MXNET_CONV_DGRAD_AS_GEMM=1

export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="0"
export DALI_ROI_DECODE="1"  #needs to be set to 1 as default and proof perf uplift

## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=20
export NHWC_BATCHNORM_LAUNCH_MARGIN=32
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25
export MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1

export NCCL_MAX_RINGS=8
export NCCL_BUFFSIZE=2097152
export NCCL_NET_GDR_READ=1     

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:30:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXHT=2         # HT is on is 2, HT off is 1
