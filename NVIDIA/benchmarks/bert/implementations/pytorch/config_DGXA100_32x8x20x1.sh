## DL params
export BATCHSIZE=20
export LR=1.7e-3
export GRADIENT_STEPS=1
export MAX_STEPS=990
export WARMUP_PROPORTION=0.0
export OPT_LAMB_BETA_1=0.87
export OPT_LAMB_BETA_2=0.97
export PHASE=2
export MAX_SAMPLES_TERMINATION=6500000
export EXTRA_PARAMS=""

## System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:40:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
