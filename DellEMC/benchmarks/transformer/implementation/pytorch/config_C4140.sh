## DL params
export MAX_TOKENS=15360
export LEARNING_RATE="1.976e-3"
export WARMUP_UPDATES=2000
#export EXTRA_PARAMS="--max-source-positions 64 --max-target-positions 64 --enable-parallel-backward-allred-opt --parallel-backward-allred-opt-threshold 105404416 --parallel-backward-allred-cuda-nstreams 2 --adam-betas (0.9,0.98) "
export EXTRA_PARAMS="--distributed-weight-update 2 --dwu-num-blocks 4 --dwu-num-rs-pg 2 --dwu-num-ar-pg 2 --dwu-num-ag-pg 0 --dwu-overlap-reductions --dwu-num-chunks 1 --dwu-flat-mt --dwu-compute-L2-grad-norm --max-source-positions 64 --max-target-positions 64 --adam-betas (0.9,0.98) "

## System run parms
export NNODES=1
#export SYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=02:00:00

## System config params
export NGPU=4
export SOCKETCORES=20
export NSOCKET=2
export HT=1         # HT is on is 2, HT off is 1
export NCCL_SOCKET_IFNAME=
