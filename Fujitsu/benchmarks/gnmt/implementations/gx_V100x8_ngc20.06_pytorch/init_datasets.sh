#!/bin/bash

DATADIR=${DATADIR:-"/data/WMT"}

bash download_dataset.sh $DATADIR
DATADIR=$DATADIR bash verify_dataset.sh
