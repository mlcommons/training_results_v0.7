#!/bin/bash

BENCHMARK=${BENCHMARK:-"rnn_translator"}
DATADIR=${DATADIR:-"/data/WMT"} # there should be ./coco2017 and ./torchvision dirs in here
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-10} # Default number of times to run the benchmark
PULL=${PULL:-0}

cd ../implementation_closed
LOGDIR=$LOGDIR DATADIR=$DATADIR BENCHMARK=$BENCHMARK PULL=$PULL NEXP=$NEXP ./run.sub
