#!/bin/bash

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

# run benchmark
set -x
DATASET_DIR='/data'
PREPROC_DATADIR='/preproc_data'
RESULTS_DIR='gnmt_wmt16'

LR=${LR:-"2.0e-3"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
WARMUP_STEPS=${WARMUP_STEPS:-200}
REMAIN_STEPS=${REMAIN_STEPS:-10336}
DECAY_INTERVAL=${DECAY_INTERVAL:-1296}
TARGET=${TARGET:-24.0}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-75}
NUMEPOCHS=${NUMEPOCHS:-8}
DIST_OPTS=${DIST_OPTS:-""}
EXTRA_OPTS=${EXTRA_OPTS:-""}
MATH=${MATH:-fp16}

PGSYSTEM=${PGSYSTEM:-"PG"}
if [[ -f config_${PGSYSTEM}.sh ]]; then
	source config_${PGSYSTEM}.sh
else
	source config_PG.sh
	echo "Unknown system, assuming PG"
fi

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' '--cpu=exclusive' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( 'python' '-u' '-m' 'bind_launch' "--nsockets_per_node=${PGNSOCKET}" \
    "--ncores_per_socket=${PGSOCKETCORES}" "--nproc_per_node=${PGNGPU}" )
fi

echo "running benchmark"

# run training
"${CMD[@]}" train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --preproc-data-dir ${PREPROC_DATADIR} \
  --target-bleu $TARGET \
  --epochs "${NUMEPOCHS}" \
  --math ${MATH} \
  --max-length-train ${MAX_SEQ_LEN} \
  --print-freq 10 \
  --train-batch-size $TRAIN_BATCH_SIZE \
  --test-batch-size $TEST_BATCH_SIZE \
  --optimizer FusedAdam \
  --lr $LR \
  --warmup-steps $WARMUP_STEPS \
  --remain-steps $REMAIN_STEPS \
  --decay-interval $DECAY_INTERVAL \
  $DIST_OPTS \
  $EXTRA_OPTS ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="RNN_TRANSLATOR"

echo "RESULT,$result_name,,$result,Fujitsu,$start_fmt"

