#/usr/bin/bash

echo 3 > /proc/sys/vm/drop_caches 

RANDOM_SEED=`date +%s`

QUALITY=0.759

set -e

# Register the model as a source root

export PYTHONPATH="$(pwd):$(pwd)/../:${PYTHONPATH}"

# MLPerf

export PYTHONPATH="$(pwd)/../logging:${PYTHONPATH}"

echo $PYTHONPATH

mkdir -p /IntelMLPerf/

MODEL_DIR="/IntelMLPerf/resnet_imagenet_${RANDOM_SEED}"

export OMP_NUM_THREADS=12

export KMP_BLOCKTIME=1

mpirun --allow-run-as-root -n 16  --map-by ppr:2:socket:pe=14 python imagenet_main.py $RANDOM_SEED --data_dir /root/training/imagenet_dataset  --model_dir $MODEL_DIR --train_epochs 44 --stop_threshold $QUALITY --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 2 --use_bfloat16  --enable_lars --label_smoothing=0.1 --weight_decay=0.0002  2>&1 |tee lars-44epochs_eval_every_4_epochs_${RANDOM_SEED}.log

