#!/usr/bin/env bash

BASE_LOG_DIR=tests/results

for d in ${BASE_LOG_DIR}/* ; do
    _dirname=$(basename "${d}")
    tests/generate_report.py --input-dir $d --output report_${_dirname}.html
done
