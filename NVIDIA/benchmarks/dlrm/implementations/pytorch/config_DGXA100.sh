## DL params
export BATCH_SIZE=55296
export LR_ARGS="--lr 24 --warmup_steps 2750 --decay_end_lr 0 --decay_steps 27772 --decay_start_step 49315"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00
export OMPI_MCA_btl="^openib"
export DGXNGPU=8
export MOUNTS=/lustre:/lustre,/raid:/raid
