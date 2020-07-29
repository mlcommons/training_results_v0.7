## DL params
export EXTRA_PARAMS='--batch-size=56 --lr-warmup-epoch=5 --lr=3.2e-3 --weight-decay=1.3e-4 --dali-workers=3'

## System run parms
export NNODES=4
#export SYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:30:00

## System config params
export NGPU=4
export SOCKETCORES=20
export NSOCKET=2
export HT=1         # HT is on is 2, HT off is 1
