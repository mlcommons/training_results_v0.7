#!/bin/bash

BENCHMARK=${BENCHMARK:-"image_classification"}

cd ../implementation_closed
docker build --pull -t mlperf-fujitsu:$BENCHMARK .
