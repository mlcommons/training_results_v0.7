#!/bin/bash

BENCHMARK=${BENCHMARK:-"rnn_translator"}

docker build --pull -t mlperf-inspur:$BENCHMARK .
