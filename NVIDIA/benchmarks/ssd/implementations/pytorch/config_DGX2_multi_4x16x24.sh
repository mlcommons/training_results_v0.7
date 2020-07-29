## DL params
export EXTRA_PARAMS='--batch-size=24 --eval-batch-size=40 --warmup=850 --num-workers=3 --lr=2.9e-3 --wd=1.7e-4 --use-nvjpeg'

## System run parms
export DGXNNODES=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2 	# HT is on is 2, HT off is 1
