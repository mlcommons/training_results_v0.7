#!/bin/bash

BENCHMARK=${BENCHMARK:-"recommendation"}

docker build --pull -t mlperf-inspur:$BENCHMARK .
