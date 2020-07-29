#!/bin/bash
##mlperf only
##real world container or virutal machine only require numer of GPUs

DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S'`}

## Data, container and volumes
BENCHMARK=${BENCHMARK:-"image_classification"}
BENCHMARK_NAME="resnet"
CONT=${CONT:-"mlperf-alibaba:$BENCHMARK"}
DATADIR=${DATADIR:-"/data"}
LOGDIR=${LOGDIR:-"/log/$BENCHMARK"}

## Load system-specific parameters for benchmark
source config_PAI.sh

## Check whether we are running in a slurm env
# Create results directory
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

##the runscripts on eflops
mpirun --allow-run-as-root --hostfile machines -np $1 -bind-to none -map-by slot --mca btl tcp,self,vader  -mca pml ob1 -x LD_LIBRARY_PATH -x PATH /workspace/mlperf/run_and_time.sh
~
