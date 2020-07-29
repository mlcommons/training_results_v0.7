#!/bin/bash

BENCHMARK=${BENCHMARK:-"translation"}

docker build --pull -t mlperf-inspur:$BENCHMARK .
