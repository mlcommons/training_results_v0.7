## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=2

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=23:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXHT=2 	# HT is on is 2, HT off is 1

## Data mount location
export DATADIR=/raid/local/mlperf/datasets/minigo_data_19x19/

## supress Tensorflow messages
## 3->FATAL, 2->ERROR, 1->WARNING, 0
export TF_CPP_MIN_LOG_LEVEL=3

## Benchmark knobs for this config.
export NUM_GPUS_TRAIN=4
export NUM_ITERATIONS=70

#selfplay perf. params
export SP_THREADS=6
export PA_SEARCH=8
export PA_INFERENCE=2
export CONCURRENT_GAMES=32
export PROCS_PER_GPU=2
