## DL params
export EXTRA_PARAMS='--batch-size=120 --eval-batch-size=160 --warmup=650 --lr=2.92e-3 --wd=1.6e-4 --use-nvjpeg'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
