#!/bin/bash

: "${DATASET_DIR:=/datasets/ssd}"
: "${CONTAINER_IMAGE:=nvcr.io/nvidia/tensorflow:20.03-tf2-py3}"

while [ "$1" != "" ]; do
    case $1 in
        -d | --dataset-dir )    shift
                                DATASET_DIR=$1
                                ;;
        -c | --container )      shift
                                CONTAINER_IMAGE=$1
                                ;;
    esac
    shift
done

docker run -it --rm \
    --gpus=all \
    --ipc=host \
    --env PYTHONDONTWRITEBYTECODE=1 \
    -v $(pwd):/scratch \
    -v ${DATASET_DIR}:/datasets \
    ${CONTAINER_IMAGE} \
    /bin/bash -c \
    "echo 'Processing validation set:' && \
     /scratch/scripts/datasets/create_tfrecord.py \
         -i /datasets/coco2017/val2017/ \
         -a /datasets/coco2017/annotations/instances_val2017.json \
         -o /datasets/coco2017/tfrecord/val \
         --ltrb \
         --ratio \
         -n 1 && \
     echo '' && \
     echo 'Processing training set:' && \
     /scratch/scripts/datasets/create_tfrecord.py \
         -i /datasets/coco2017/train2017/ \
         -a /datasets/coco2017/annotations/instances_train2017.json \
         -o /datasets/coco2017/tfrecord/train \
         --ltrb \
         --ratio \
         --skip-empty \
         --seed 2020 \
         -n 1"
