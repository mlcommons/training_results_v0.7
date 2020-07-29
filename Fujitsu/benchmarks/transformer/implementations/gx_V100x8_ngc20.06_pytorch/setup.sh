#!/bin/bash

BENCHMARK=${BENCHMARK:-"transformer"}

cd ../implementation_closed
docker build --pull -t mlperf-fujitsu:$BENCHMARK .
