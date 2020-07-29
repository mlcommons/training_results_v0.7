#!/bin/bash

DOWNLOAD_PATH='/datasets/downloads/coco2017'
OUTPUT_PATH='/datasets/coco2017'

mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

mkdir -p $OUTPUT_PATH
unzip train2017.zip -d $OUTPUT_PATH
unzip val2017.zip -d $OUTPUT_PATH
unzip annotations_trainval2017.zip -d $OUTPUT_PATH
