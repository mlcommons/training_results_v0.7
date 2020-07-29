## DL params
export CONFIG="mlperf_fp16_dgxa100.json"
export BATCH_SIZE=65536

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00
export OMPI_MCA_btl="^openib"
export DGXNGPU=8
export MOUNTS=/raid:/raid

## NCCL WAR
export NCCL_LAUNCH_MODE=PARALLEL
