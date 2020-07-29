#!/bin/bash

BENCHMARK=${BENCHMARK:-"single_stage_detector"}

cd ../implementation_closed
docker build --pull -t mlperf-fujitsu:$BENCHMARK .
