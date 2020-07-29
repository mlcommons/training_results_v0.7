## DL params
export MAX_TOKENS=768
export LEARNING_RATE="1.732e-3"
export WARMUP_UPDATES=400
export EXTRA_PARAMS="--distributed-weight-update 2 --dwu-num-blocks 4 --dwu-num-rs-pg 2 --dwu-num-ar-pg 2 --dwu-num-ag-pg 0 --dwu-overlap-reductions --dwu-num-chunks 1 --dwu-flat-mt --dwu-compute-L2-grad-norm --max-source-positions 76 --max-target-positions 76 --adam-betas (0.86,0.92) "

## System run parms
export DGXNNODES=60
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:25:00

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2 	# HT is on is 2, HT off is 1
