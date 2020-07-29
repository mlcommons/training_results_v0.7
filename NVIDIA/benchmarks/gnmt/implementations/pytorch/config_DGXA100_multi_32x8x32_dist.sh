## System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=${WALLTIME:-"00:10:00"}

## DL params
export LR=${LR:-"4.0e-3"}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
export TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-16}
export WARMUP_STEPS=${WARMUP_STEPS:-200}
export REMAIN_STEPS=${REMAIN_STEPS:-2259}
export DECAY_INTERVAL=${DECAY_INTERVAL:-283}
export TARGET=${TARGET:-24.0}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
export NUMEPOCHS=${NUMEPOCHS:-12}
export MATH=${MATH:-fp16}
export DIST_OPTS=${DIST_OPTS-"\
   --distributed-weight-update 2 \
   --dwu-num-blocks 3 \
   --dwu-num-chunks 2 \
   --dwu-num-rs-pg 1 \
   --dwu-num-ar-pg 2 \
   --dwu-num-ag-pg 0 \
   --dwu-overlap-reductions \
   --dwu-grad-norm \
   "}
export EXTRA_OPTS=${EXTRA_OPTS-"\
   --fused-attention \
   --fused-xentropy \
   --prealloc-mode=once \
   --no-log-all-ranks \
   "}
export CUDA_DEVICE_MAX_CONNECTIONS=32
export OMPI_MCA_btl="^openib"
export NCCL_SOCKET_IFNAME="ib0"
export NCCL_TOPO_FILE="/workspace/rnn_translator/dgxa100_nic_affinity.xml"


## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXHT=2 	# HT is on is 2, HT off is 1
export DGXNSOCKET=2
