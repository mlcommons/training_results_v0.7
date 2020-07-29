## DL params
export EXTRA_PARAMS='--batch-size=56 --lr-warmup-epoch=5 --lr=3.2e-3 --weight-decay=1.3e-4 --dali-workers=3 --precision=fp32 --data-layout=NCHW'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:00:00

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
