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
#   run_and_time_multi.sh

set -e

# GPU driver path
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH

# container wise env vars
HOROVOD_CYCLE_TIME=0.1                      
HOROVOD_BATCH_D2D_MEMCOPIES=1               
HOROVOD_NUM_NCCL_STREAMS=1                  
MXNET_CUDNN_AUTOTUNE_DEFAULT=0              
MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999 
MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999 
MXNET_GPU_WORKER_NTHREADS=1                 
MXNET_CPU_PRIORITY_NTHREADS=1               
MXNET_EXPERIMENTAL_ENABLE_CUDA_GRAPH=1      
OMP_NUM_THREADS=1                           
OMPI_MCA_btl=^openib                        
OPENCV_FOR_THREADS_NUM=1

cd /mnt/current
readonly _config_file="./config_${SYSTEM}.sh"
source $_config_file

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
NUMEPOCHS=${NUMEPOCHS:-80}

#echo "/tmp in container is"
#ls /tmp

echo "running benchmark"

export DATASET_DIR="/data/coco2017"
export PRETRAINED_DIR="/pretrained/mxnet"


#if [ "$PMIX_RANK" = "0" ]; then
#    echo "*****************************printing env********************************"
#    env
#    echo "***************************done printing env*****************************"
#fi

# run training
python ssd_main_async.py \
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
