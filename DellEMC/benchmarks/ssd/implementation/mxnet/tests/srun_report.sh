#!/usr/bin/env bash

srun \
    -Jreport \
    --account=mlperft-ssd \
    --container-workdir=/workspace \
    --container-mounts=$(pwd):/workspace \
    --ntasks-per-node=1 \
    -N1 \
    -t5 \
    --container-image=nvcr.io/nvidian/swdl/akiswani_ssd_test_report:latest \
    tests/generate_reports.sh &
