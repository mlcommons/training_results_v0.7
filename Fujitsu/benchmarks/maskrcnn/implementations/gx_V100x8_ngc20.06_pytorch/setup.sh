#!/bin/bash

BENCHMARK=${BENCHMARK:-"object_detection"}

cd ../implementation_closed
docker build --pull -t mlperf-fujitsu:$BENCHMARK .
