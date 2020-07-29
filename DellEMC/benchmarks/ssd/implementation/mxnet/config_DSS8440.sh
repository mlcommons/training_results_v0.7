## DL params
export EXTRA_PARAMS='--batch-size=128 --lr-warmup-epoch=5 --lr=5.84e-3 --weight-decay=1.6e-4' #Parameters from 2xC4140 exact same 25.00

## System run parms
export NNODES=1
#export SYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#WALLTIME_MINUTES=30
#export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))
export WALLTIME=00:30:00

## System config params
export NGPU=8
export SOCKETCORES=20
export NSOCKET=2
export HT=2         # HT is on is 2, HT off is 1

export TOBIND=0
export NCCL_SOCKET_IFNAME=
