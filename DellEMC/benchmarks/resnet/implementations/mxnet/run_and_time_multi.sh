#!/bin/bash
#
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time_multi.sh

# GPU driver path
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH

# container wise env vars
export MXNET_UPDATE_ON_KVSTORE=0      
export MXNET_EXEC_ENABLE_ADDTO=1      
export MXNET_USE_TENSORRT=0           
export MXNET_GPU_WORKER_NTHREADS=1    
export MXNET_GPU_COPY_NTHREADS=1      
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0 
export NCCL_BUFFSIZE=2097152          
export NCCL_NET_GDR_READ=1            
export HOROVOD_BATCH_D2D_MEMCOPIES=1  
export HOROVOD_GROUPED_ALLREDUCES=1  
export NCCL_SOCKET_IFNAME=ib0 
export OMP_NUM_THREADS=1 
export OPENCV_FOR_THREADS_NUM=1


cd /mnt/current
readonly _config_file="./config_${SYSTEM}.sh"
source $_config_file

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
OPTIMIZER=${OPTIMIZER:-"sgd"}
BATCHSIZE=${BATCHSIZE:-1664}
KVSTORE=${KVSTORE:-"device"}
LR=${LR:-"0.6"}
MOM=${MOM:-"0.9"}
LRSCHED=${LRSCHED:-"30,60,80"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LARSETA=${LARSETA:-'0.001'}
DALI_HW_DECODER_LOAD=${DALI_HW_DECODER_LOAD:-'0.0'}
WD=${WD:-'0.0001'}
LABELSMOOTHING=${LABELSMOOTHING:-'0.0'}
SEED=${SEED:-1}
EVAL_OFFSET=${EVAL_OFFSET:-2}
EVAL_PERIOD=${EVAL_PERIOD:-4}
DALI_PREFETCH_QUEUE=${DALI_PREFETCH_QUEUE:-2}
DALI_NVJPEG_MEMPADDING=${DALI_NVJPEG_MEMPADDING:-64}
DALI_THREADS=${DALI_THREADS:-3}
DALI_CACHE_SIZE=${DALI_CACHE_SIZE:-0}
DALI_ROI_DECODE=${DALI_ROI_DECODE:-0}
DALI_DONT_USE_MMAP=${DALI_DONT_USE_MMAP:-0}
NUMEPOCHS=${NUMEPOCHS:-90}
NETWORK=${NETWORK:-"resnet-v1b-fl"}
BN_GROUP=${BN_GROUP:-1}
PROFILE=${PROFILE:-0}
NODALI=${NODALI:-0}
NUMEXAMPLES=${NUMEXAMPLES:-}
PROFILE_ALL_LOCAL_RANKS=${PROFILE_ALL_LOCAL_RANKS:-0}
THR="0.759"


DATAROOT="/data"

echo "running benchmark"
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}


GPUS=$(seq 0 $(($NGPU - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
      --gpus               "${GPUS}"
      --batch-size         "${BATCHSIZE}"
      --kv-store           "${KVSTORE}"
      --lr                 "${LR}"
      --mom                "${MOM}"
      --lr-step-epochs     "${LRSCHED}"
      --lars-eta           "${LARSETA}"
      --label-smoothing    "${LABELSMOOTHING}"
      --wd                 "${WD}"
      --warmup-epochs      "${WARMUP_EPOCHS}"
      --eval-period        "${EVAL_PERIOD}"
      --eval-offset        "${EVAL_OFFSET}"
      --optimizer          "${OPTIMIZER}"
      --network            "${NETWORK}"
      --num-layers         "50"
      --num-epochs         "${NUMEPOCHS}"
      --accuracy-threshold "${THR}"
      --seed               "${SEED}"
      --dtype              "float16"
      --disp-batches       "20"
      --image-shape        "4,224,224"
      --fuse-bn-relu       "1"
      --fuse-bn-add-relu   "1"
      --bn-group           "${BN_GROUP}"
      --min-random-area    "0.05"
      --max-random-area    "1.0"
      --conv-algo          "1"
      --force-tensor-core  "1"
      --input-layout       "NHWC"
      --conv-layout        "NHWC"
      --batchnorm-layout   "NHWC"
      --pooling-layout     "NHWC"
      --batchnorm-mom      "0.9"
      --batchnorm-eps      "1e-5"
      --data-train         "${DATAROOT}/train.rec"
      --data-train-idx     "${DATAROOT}/train.idx"
      --data-val           "${DATAROOT}/val.rec"
      --data-val-idx       "${DATAROOT}/val.idx"
      --dali-dont-use-mmap "${DALI_DONT_USE_MMAP}"
      --dali-hw-decoder-load "${DALI_HW_DECODER_LOAD}"
      --dali-prefetch-queue        "${DALI_PREFETCH_QUEUE}"
      --dali-nvjpeg-memory-padding "${DALI_NVJPEG_MEMPADDING}"
      --dali-threads       "${DALI_THREADS}"
      --dali-cache-size    "${DALI_CACHE_SIZE}"
      --dali-roi-decode    "${DALI_ROI_DECODE}"
      --profile            "${PROFILE}"
)
if [[ ${NODALI} -lt 1 ]]; then
    PARAMS+=(
    --use-dali 
    )
fi

# If numexamples is set then we will override the numexamples
if [[ ${NUMEXAMPLES} -ge 1 ]]; then
        PARAMS+=(
        --num-examples "${NUMEXAMPLES}"
        )
fi

python train_imagenet.py "${PARAMS[@]}"; ret_code=$?

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
export PROFILE=0
