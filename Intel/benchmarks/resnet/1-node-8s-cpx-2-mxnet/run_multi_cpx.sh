#!/bin/bash
workspace=$(pwd)
func_type=$1
hostfile=$2

node_count=$(cat $hostfile | wc -l)
num_sockets=$(lscpu | grep 'Socket(s)' | awk '{print$2}')
ppn=$num_sockets
# num_proc=$[${node_count} * ${ppn}]
num_proc=32
ppn=32

phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')
# omp_threads=$[${phy_cpus}-2]
omp_threads=5

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# clear cache
sync; echo 3 > /proc/sys/vm/drop_caches

# Random number seed
seed=${start}

MXNET_ROOT=$3
DATA_ROOT=$4

if [ -z $MXNET_ROOT ];then
    echo "Please specify the root directory of MXNet as MXNET_ROOT."
    exit 1
fi

if [ -z $DATA_ROOT ];then
    echo "Please specify the root directory of ImageNet dataset as DATA_ROOT."
    exit 1
fi

if [[ -n $I_MPI_ROOT && -z $MLSL_ROOT ]];then
    echo "Error: must source MLSL env first and then source MPI env."
    exit 1
fi

if [ -n $MLSL_ROOT ];then
    echo "Warning: please make sure that you have sourced MLSL env first before sourcing MPI env in case that you used the incorrect I_MPI_ROOT."
fi

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

# launch FP32/BFP16 training
# running 4 instances (1 ins per socket) on CPX machine with 24 physical cores.
${I_MPI_ROOT}/intel64/bin/mpirun \
    -n $num_proc \
    -ppn $ppn \
    -f $hostfile \
    -genv MLSL_LOG_LEVEL=0 \
    -genv I_MPI_DEBUG=0 \
    -genv I_MPI_PIN_DOMAIN=7:compact \
    -genv OMP_NUM_THREADS=$omp_threads \
    -genv MXNET_USE_OPERATOR_TUNING=0 \
    -genv HOROVOD_FUSION_THRESHOLD=0 \
    ./run.sh $func_type $MXNET_ROOT $DATA_ROOT $node_count $num_proc $seed
# [0x000000000000000000fffffc,0x000000000000fffffc000000,0x000000fffffc000000000000,0xfffffc000000000000000000]
