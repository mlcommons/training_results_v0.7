## DL params
export EXTRA_PARAMS='--batch-size=56 --lr-warmup-epoch=5.25 --lr=2.92e-3 --weight-decay=1.6e-4'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=30
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
