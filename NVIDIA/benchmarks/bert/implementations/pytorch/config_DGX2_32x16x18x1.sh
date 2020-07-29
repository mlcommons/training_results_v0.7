## DL params
export BATCHSIZE=18
export LR=2.0e-3
export GRADIENT_STEPS=1
export MAX_STEPS=750
export WARMUP_PROPORTION=0.0
export OPT_LAMB_BETA_1=0.86
export OPT_LAMB_BETA_2=0.975
export PHASE=2
export MAX_SAMPLES_TERMINATION=7500000
export EXTRA_PARAMS=""

## System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:30:00

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
