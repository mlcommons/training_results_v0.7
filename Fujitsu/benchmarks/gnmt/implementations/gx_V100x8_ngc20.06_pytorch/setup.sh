#!/bin/bash

BENCHMARK=${BENCHMARK:-"rnn_translator"}

cd ../implementation_closed
docker build --pull -t mlperf-fujitsu:$BENCHMARK .
