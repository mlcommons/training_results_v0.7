#!/bin/bash

BENCHMARK=${BENCHMARK:-transformer}

cd ../implementation_closed
nvidia-docker run --rm --privileged --ipc=host -v $PWD:/workspace/translation mlperf-fujitsu:$BENCHMARK /workspace/translation/data_create.sh
