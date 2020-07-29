#!/bin/bash

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

if [[ ${WORLD_SIZE:-${SLURM_NTASKS}} -ne 1 ]]; then
    DISTRIBUTED_INIT_METHOD="--distributed-init-method env://"
else
    DISTRIBUTED_INIT_METHOD="--distributed-world-size 1"
fi

# These are scanned by train.py, so make sure they are exported
export DGXSYSTEM
export SLURM_NTASKS_PER_NODE
export SLURM_NNODES

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
  CMD=( 'python' '-u' '-m' 'bind_launch' "--nsockets_per_node=${DGXNSOCKET}" \
    "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
fi

"${CMD[@]}" train.py ${DATASET_DIR} \
  --seed ${SEED} \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm "0.0" \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr "0.0" \
  --warmup-updates ${WARMUP_UPDATES} \
  --lr ${LEARNING_RATE} \
  --min-lr "0.0" \
  --dropout "0.1" \
  --weight-decay "0.0" \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing "0.1" \
  --max-tokens ${MAX_TOKENS} \
  --max-epoch ${NUMEPOCHS} \
  --target-bleu "25.0" \
  --ignore-case \
  --no-save \
  --update-freq 1 \
  --fp16 \
  --seq-len-multiple 2 \
  --source_lang en \
  --target_lang de \
  --bucket_growth_factor 1.035 \
  --batching_scheme "v0p5_better" \
  --batch_multiple_strategy "dynamic" \
  --fast-xentropy \
  --max-len-a 1 \
  --max-len-b 50 \
  --lenpen 0.6 \
  --no-progress-bar \
  --dataloader-num-workers 2 \
  --enable-dataloader-pin-memory \
  --multihead-attn-impl 'fast_with_lyrnrm_and_dropoutadd' \
  ${DISTRIBUTED_INIT_METHOD} \
  ${EXTRA_PARAMS} ; ret_code=$?

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="transformer"
echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"

