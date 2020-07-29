## System run parms
export NFNNODES=1
export NFSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=${WALLTIME:-"00:15:00"}

## DL params
export LR=${LR:-"2.875e-3"}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-384}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
export WARMUP_STEPS=${WARMUP_STEPS:-200}
export REMAIN_STEPS=${REMAIN_STEPS:-4054}
export DECAY_INTERVAL=${DECAY_INTERVAL:-506}
export TARGET=${TARGET:-24.0}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
export NUMEPOCHS=${NUMEPOCHS:-8}
export MATH=${MATH:-fp16}
export DIST_OPTS=${DIST_OPTS-"\
   --distributed-weight-update 2 \
   --dwu-num-blocks 1 \
   --dwu-num-chunks 2 \
   --dwu-num-rs-pg 2 \
   --dwu-num-ar-pg 2 \
   --dwu-num-ag-pg 0 \
   --dwu-grad-norm \
   "}
export EXTRA_OPTS=${EXTRA_OPTS-"\
   --fused-attention \
   --fused-xentropy \
   --no-log-all-ranks \
   "}

## System config params
export NFNGPU=8
export NFSOCKETCORES=64
export NFHT=2 	# HT is on is 2, HT off is 1
export NFNSOCKET=2
