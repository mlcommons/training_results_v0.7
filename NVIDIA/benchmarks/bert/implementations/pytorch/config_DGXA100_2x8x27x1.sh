## DL params
export BATCHSIZE=27
export LR=4.0e-4
export GRADIENT_STEPS=1
export MAX_STEPS=8103
export WARMUP_PROPORTION=0.0
export PHASE=2
export MAX_SAMPLES_TERMINATION=4500000
export EXTRA_PARAMS="--unpad"

## System run parms
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=02:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
