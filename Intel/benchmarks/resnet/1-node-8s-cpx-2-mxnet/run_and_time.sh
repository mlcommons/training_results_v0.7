#!/bin/bash

# clear cache
echo 1 > /proc/sys/vm/compact_memory; sleep 1
echo 3 > /proc/sys/vm/drop_caches; sleep 1

export PYTHONPATH=$(pwd)/resnet/incubator-mxnet/python
source $(pwd)/resnet/mlsl/_install/intel64/bin/mlslvars.sh thread
source $HOME/intel/impi/2019.5.281/intel64/bin/mpivars.sh release_mt

if [ -z $DATA_ROOT ];then
    echo "Please specify the root directory of ImageNet dataset as DATA_ROOT."
    exit 1
fi

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
seed=${start}
host_name=$(hostname)

export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
${I_MPI_ROOT}/intel64/bin/mpirun -n 32 -ppn 32 -localhost ${host_name} -genv I_MPI_PIN_DOMAIN=7:compact -genv OMP_NUM_THREADS=5 -genv HOROVOD_FUSION_THRESHOLD=0 ./run.sh resnet50-v1.5_multi_ins_hyve_bf16 $(pwd)/resnet/incubator-mxnet $DATA_ROOT 1 32 $seed 2>&1 | tee mxnet_resnet50_8_sockets_${seed}.log

# post processing on logging
python post_processing.py --in-file mxnet_resnet50_8_sockets_${seed}.log --out-file result_0.txt