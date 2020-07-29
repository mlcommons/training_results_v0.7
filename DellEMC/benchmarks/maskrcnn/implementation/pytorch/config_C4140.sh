## DL params
export EXTRA_PARAMS=""
export EXTRA_CONFIG='SOLVER.BASE_LR 0.03 SOLVER.MAX_ITER 160000 SOLVER.WARMUP_FACTOR 0.000048 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (48000,64000) SOLVER.IMS_PER_BATCH 24 TEST.IMS_PER_BATCH 4 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC True'

## System run parms
export NNODES=1
export SYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
export NGPU=4
export SOCKETCORES=20
export NSOCKET=2
export HT=1         # HT is on is 2, HT off is 1
export NCCL_SOCKET_IFNAME=
