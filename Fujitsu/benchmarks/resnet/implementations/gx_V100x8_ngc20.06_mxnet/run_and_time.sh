#!/bin/bash

BENCHMARK=${BENCHMARK:-"image_classification"}
DATADIR=${DATADIR:-"data/imagenet/train-val-recordio-passthrough"} 
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-5} # Default number of times to run the benchmark
PULL=${PULL:-0}

cd ../implementation_closed
LOGDIR=$LOGDIR DATADIR=$DATADIR BENCHMARK=$BENCHMARK PULL=$PULL NEXP=$NEXP ./run.sub
