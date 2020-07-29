#!/bin/bash
set -e 
set -o pipefail

TF_INSTALLATION_PATH=$(python -c "import tensorflow as tf, os; print(os.path.dirname(tf.__file__))")
cp lars_optimizer.py $TF_INSTALLATION_PATH/python/training
cd $TF_INSTALLATION_PATH/python/training
sed -i "81ifrom tensorflow.python.training.lars_optimizer import LARSOptimizer" $TF_INSTALLATION_PATH/_api/v2/compat/v1/train/__init__.py
