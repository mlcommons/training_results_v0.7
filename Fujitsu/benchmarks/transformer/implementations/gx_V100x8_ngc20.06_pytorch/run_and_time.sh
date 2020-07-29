#!/bin/bash

BENCHMARK=${BENCHMARK:-"transformer"}
DATADIR=${DATADIR:-"/implementation_closed/examples/translation/wmt14_en_de/utf8"} 
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-10} # Default number of times to run the benchmark
PULL=${PULL:-0}

cd ../implementation_closed
LOGDIR=$LOGDIR DATADIR=$DATADIR BENCHMARK=$BENCHMARK PULL=$PULL NEXP=$NEXP ./run.sub
