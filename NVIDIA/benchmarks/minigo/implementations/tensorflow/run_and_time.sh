#!/bin/bash

#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DGXSYSTEM=${DGXSYSTEM:-"DGX1"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX2_multi.sh
  echo "Unknown system, assuming DGX1"
fi

USE_TRT=${USE_TRT:-"1"}
VERBOSE=${VERBOSE:-"0"}
NUM_GPUS_TRAIN=${NUM_GPUS_TRAIN:-$DGXNGPU}
NUM_ITERATIONS=${NUM_ITERATIONS:-"75"}
SUGGESTED_GAMES=${SUGGESTED_GAMES:-"8192"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-"4096"}
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}

# selfplay perf. params
PROCS_PER_GPU=${PROCS_PER_GPU:-"1"}
SP_THREADS=${SP_THREADS:-"6"}
PA_SEARCH=${PA_SEARCH:-"8"}
PA_INFERENCE=${PA_INFERENCE:-"2"}
CONCURRENT_GAMES=${CONCURRENT_GAMES:-"32"}
SLURM_JOB_ID=${SLURM_JOB_ID:-"000786"}
SLURM_NODEID=${SLURM_NODEID:-"0"}
SLURM_LOCALID=${SLURM_LOCALID:-"0"}

# avoid mpi warnings
export OMPI_MCA_mpi_warn_on_fork=0
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0

# avoid TF Deprecation not enabled warning
export TF_ENABLE_DEPRECATION_WARNINGS=1

# print command only for rank-0
if [ "${SLURM_NODEID}" -eq 0 ] && [ "${SLURM_LOCALID}" -eq 0 ] ;
then
    echo "Run vars: id $SLURM_JOB_ID gpus $DGXNGPU tasks/node $SLURM_NTASKS_PER_NODE procs/gpu $PROCS_PER_GPU"
    echo "Running benchmark REINFORCEMENT - Minigo"
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
    set -x
fi

set -e
cd minigo

BASE_DIR=ml_perf/mpi-results/slurm-$SLURM_JOB_ID
CHECKPOINT_DIR="/data/mlperf07"
TARGET_PATH="/data/target/target.minigo.tf"

python3 ./ml_perf/train_loop.py \
        --board_size=19 \
        --base_dir=$BASE_DIR \
        --flagfile=ml_perf/flags/19/train_loop.flags \
        --checkpoint_dir=$CHECKPOINT_DIR \
        --target_path=$TARGET_PATH \
        --num_gpus_train=$NUM_GPUS_TRAIN \
        --ranks_per_node=$SLURM_NTASKS_PER_NODE \
        --procs_per_gpu=$PROCS_PER_GPU \
        --use_trt=$USE_TRT \
        --verbose=$VERBOSE \
        --selfplay_threads=$SP_THREADS \
        --parallel_search=$PA_SEARCH \
        --parallel_inference=$PA_INFERENCE \
        --concurrent_games_per_thread=$CONCURRENT_GAMES \
        --train_batch_size=$TRAIN_BATCH_SIZE \
        --suggested_games_per_iteration=$SUGGESTED_GAMES \
        --iterations=$NUM_ITERATIONS


# print command only for rank-0
if [ "${SLURM_NODEID}" -eq 0 ] && [ "${SLURM_LOCALID}" -eq 0 ] ;
then
    set +x
    echo "Done with benchmark REINFORCEMENT - Minigo"
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
fi
