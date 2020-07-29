## DL params
# steps500_lr0.004_wup0.1_b1_0.878_b2_0.974
export BATCHSIZE=4
export LR=0.00288293
export GRADIENT_STEPS=1
export MAX_STEPS=560
export WARMUP_STEPS=287
export START_WARMUP_STEP=-76
export OPT_LAMB_BETA_1=0.88
export OPT_LAMB_BETA_2=0.88
export TARGET_MLM_ACCURACY=0.706
export WEIGHT_DECAY_RATE=0.0166629
export PHASE=2
export MAX_SAMPLES_TERMINATION=12000000
export INIT_LOSS_SCALE=16384
export EVAL_ITER_START_SAMPLES=3047424
export EVAL_ITER_SAMPLES=507904
export EXTRA_PARAMS=" "

## System run parms
export DGXNNODES=256
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:50:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1
