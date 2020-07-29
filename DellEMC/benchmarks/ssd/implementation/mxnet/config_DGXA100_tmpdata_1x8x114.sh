# this config gets data from a tar file and puts it in /tmp in the container
export CREATE_TMP_FILE_FROM="/lustre/fsw/mlperft-ssd/datasets/pytorch-coco2017.tar"

## DL params
export EXTRA_PARAMS='--batch-size=114 --lr-warmup-epoch=5 --lr=3.2e-3 --weight-decay=1.3e-4'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:15:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

export NUMEPOCHS=10
