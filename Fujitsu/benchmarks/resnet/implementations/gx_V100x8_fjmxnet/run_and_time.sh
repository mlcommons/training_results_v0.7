#!/bin/bash

DATADIR=${DATADIR:-"data/imagenet/train-val-recordio-passthrough"} 
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-5} # Default number of times to run the benchmark
PULL=${PULL:-0}

cd ../implementation_open/mlperf_implementations
LOGDIR=$LOGDIR DATADIR=$DATADIR BENCHMARK=$BENCHMARK NEXP=$NEXP ./gx_run.sub
