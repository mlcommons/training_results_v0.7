#!/usr/bin/env bash
source ./scripts/docker/config.sh


DATA_DIR=$1
RESULTS_DIR=$2

docker run -it --rm \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --gpus=all \
  --ipc=host \
  -v $PWD:/workspace/ssd \
  -v "$DATA_DIR":/datasets \
  -v "$RESULTS_DIR":/results \
  $target_docker_image bash
