## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=2

## System run parms
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2 	# HT is on is 2, HT off is 1

## Data mount location
export DATADIR=/lustre/fsr/datasets/minigo_data_19x19/

## supress Tensorflow messages
## 3->FATAL, 2->ERROR, 1->WARNING, 0
export TF_CPP_MIN_LOG_LEVEL=3

## Benchmark knobs for this config.
export NUM_GPUS_TRAIN=4
export NUM_ITERATIONS=65

#multiple procs/gpu
export SP_THREADS=2
export PA_SEARCH=2
export PA_INFERENCE=2
export CONCURRENT_GAMES=32
export PROCS_PER_GPU=4

#
export OMPI_MCA_btl="^openib"
export NCCL_SOCKET_IFNAME=ib0
