## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=2

## System run parms
export DGXNNODES=48
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:30:00

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXHT=2 	# HT is on is 2, HT off is 1

## Turn on FP16 training via AMP
export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
export TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1

## supress Tensorflow messages
## 3->FATAL, 2->ERROR, 1->WARNING, 0
export TF_CPP_MIN_LOG_LEVEL=3

## Benchmark knobs for this config.
export NUM_GPUS_TRAIN=64
export NUM_ITERATIONS=95

## no TRT at scale
export USE_TRT=0

#selfplay perf. params
export SP_THREADS=3
export PA_SEARCH=6
export PA_INFERENCE=2
export CONCURRENT_GAMES=64
export PROCS_PER_GPU=1
