# this config gets data from a tar file and puts it in /tmp in the container
export CREATE_TMP_FILE_FROM="/home/mfrank/work/jun/pytorch-coco2017.tar"

## DL params
export EXTRA_PARAMS='--batch-size=56 --lr-warmup-epoch=5.25 --lr=2.92e-3 --weight-decay=1.6e-4'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:10:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

export NUMEPOCHS=10
