## DL params
export CONFIG="mlperf_fp16_dgx2_16gpu.json"
export BATCH_SIZE=65536

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00
export OMPI_MCA_btl="^openib"
export DGXNGPU=16
export MOUNTS=/raid/datasets:/raid/datasets,/gpfs/fs1:/gpfs/fs1
