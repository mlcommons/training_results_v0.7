#!/bin/bash

DOWNLOAD_PATH='/datasets/downloads/coco2014'
OUTPUT_PATH='/datasets/coco2014'

mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

mkdir -p $OUTPUT_PATH
unzip train2014.zip -d $OUTPUT_PATH
unzip val2014.zip -d $OUTPUT_PATH
unzip annotations_trainval2014.zip -d $OUTPUT_PATH
