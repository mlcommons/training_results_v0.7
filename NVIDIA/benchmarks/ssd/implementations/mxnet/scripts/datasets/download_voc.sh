#!/bin/bash

DOWNLOAD_PATH='/datasets/downloads/voc'
OUTPUT_PATH='/datasets/voc'

mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

mkdir -p $OUTPUT_PATH
tar -xvf VOCtrainval_11-May-2012.tar -C $OUTPUT_PATH
tar -xvf VOCtrainval_06-Nov-2007.tar -C $OUTPUT_PATH
tar -xvf VOCtest_06-Nov-2007.tar -C $OUTPUT_PATH
