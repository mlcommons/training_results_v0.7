#!/bin/bash

# GPU driver path
export PATH=/mnt/driver/bin:$PATH
export LD_LIBRARY_PATH=/mnt/driver/lib64:$LD_LIBRARY_PATH

cd /mnt/current
pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
pip install --no-cache-dir -r requirements.txt

python setup.py install.
/opt/conda/bin/conda install -y numpy==1.17.4

