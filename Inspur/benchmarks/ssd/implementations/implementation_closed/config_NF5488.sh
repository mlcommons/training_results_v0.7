## DL params
export EXTRA_PARAMS='--batch-size=114 --warmup=650 --lr=3.2e-3 --wd=1.3e-4 --num-workers=8'

## System run parms
export NFNNODES=1
export NFSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export NEXP=5
WALLTIME_MINUTES=20
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export NFNGPU=8
export NFSOCKETCORES=64
export NFNSOCKET=2
export NFHT=2         # HT is on is 2, HT off is 1
