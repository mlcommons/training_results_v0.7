## DL params
# FIXME mfrank 2020-Feb-18, lr-decay-epochs to 60,75 when we fix the
# off-by-one-error in main loop
export EXTRA_PARAMS='--lr-decay-epochs 61 76 --batch-size=2 --eval-batch-size=110 --lr-warmup-epoch=26 --lr=4.57e-3 --weight-decay=4e-5 --bn-group=1 --gradient-predivide-factor=32 --input-batch-multiplier=20'

export NUMEPOCHS=10

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:15:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
