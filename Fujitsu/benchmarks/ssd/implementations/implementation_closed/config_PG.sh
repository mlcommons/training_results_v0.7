#!/bin/bash

## DL params
EXTRA_PARAMS=(
               --batch-size      "120"
               --eval-batch-size "160"
               --warmup          "650"
               --lr              "2.92e-3"
               --wd              "1.6e-4"
               --use-nvjpeg
#               --use-roi-decode
             )

## System run parms
PGNNODES=1
PGSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME=01:00:00

## System config params
PGNGPU=8
PGSOCKETCORES=24
PGNSOCKET=2
PGHT=2         # HT is on is 2, HT off is 1
PGIBDEVICES=''
