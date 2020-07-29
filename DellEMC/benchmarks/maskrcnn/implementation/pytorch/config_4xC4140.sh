## DL params
export EXTRA_PARAMS=""
export EXTRA_CONFIG='SOLVER.BASE_LR 0.12 SOLVER.MAX_ITER 42000 SOLVER.WARMUP_FACTOR 0.000192 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (12000,16000) SOLVER.IMS_PER_BATCH 96 TEST.IMS_PER_BATCH 16 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 6000 NHWC True'

## System run parms
export NNODES=4
#export SYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
export NGPU=4
export SOCKETCORES=20
export NSOCKET=2
export HT=1         # HT is on is 2, HT off is 1
