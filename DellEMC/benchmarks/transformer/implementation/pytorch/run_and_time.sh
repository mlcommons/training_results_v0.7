#!/bin/bash

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
MULTI_NODE=${MULTI_NODE:-''}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# Options

set -x

SEED=${SEED:-$RANDOM}
MAX_TOKENS=${MAX_TOKENS:-5120}
# DATASET_DIR="/data"
DATASET_DIR=${DATASET_DIR:-"/data"}
MODE=${MODE:-TRAIN}
NUMEPOCHS=${NUMEPOCHS:-30}

case "$MODE" in
  PREPROCESS) source run_preprocessing.sh;;
  TRAIN)      source run_training.sh;;
esac

set +x
