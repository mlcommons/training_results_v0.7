#!/bin/bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
NUMEPOCHS=${NUMEPOCHS:-80}

echo "running benchmark"

export DATASET_DIR="/data/coco2017"
export PRETRAINED_DIR="/pretrained/mxnet"

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
    if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
	cluster=''
	if [[ "${SYSTEM}" == DGX2* ]]; then
	    cluster='circe'
	fi
	if [[ "${SYSTEM}" == DGXA100* ]]; then
	    cluster='selene'
	    fi
	CMD=( './bind.sh' "--cluster=${cluster}" '--ib=single' '--' 'python' '-u' )
    else
	CMD=( 'python' '-u' )
    fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  BIND=''
  if [[ "${TOBIND}" == 1 ]]; then    
  	BIND="./bind.sh --cpu=exclusive"
  fi
  CMD="mpirun --allow-run-as-root --bind-to none --np ${NGPU} ${BIND} python -u "
fi

#if [ "$PMIX_RANK" = "0" ]; then
#    echo "*****************************printing env********************************"
#    env
#    echo "***************************done printing env*****************************"
#fi

# run training
${CMD} \
    ssd_main_async.py \
    --log-interval=100 \
    --coco-root=${DATASET_DIR} \
    --pretrained-backbone=${PRETRAINED_DIR}/resnet34-333f7ec4.pickle \
    --data-layout=NHWC \
    --epochs "${NUMEPOCHS}" \
    --async-val \
    --dataset-size 117266 \
    --eval-dataset-size 5000 \
    ${EXTRA_PARAMS} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,dellemc,$start_fmt"
