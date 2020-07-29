#!/bin/bash

BENCHMARK=${BENCHMARK:-"object_detection"}
DATADIR=${DATADIR:-"/data/coco-2017"} # there should be ./coco2017 and ./torchvision dirs in here
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-5} # Default number of times to run the benchmark
PULL=${PULL:-0}

cd ../implementation_closed
LOGDIR=$LOGDIR DATADIR=$DATADIR BENCHMARK=$BENCHMARK PULL=$PULL NEXP=$NEXP ./run.sub
