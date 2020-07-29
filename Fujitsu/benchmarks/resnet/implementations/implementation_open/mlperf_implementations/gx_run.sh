#!/bin/bash

source setenv
DEFAULT_NEXP=1
LOGDIR=${LOGDIR:-"LOG"}

if [ -z "$NEXP" ]; then
	NEXP=1
fi

for i in `seq ${NEXP}`                                                      
do
	DATESTAMP=`date +'%y%m%d%H%M%S%N'`

        echo Clearing cache
        sudo /sbin/sysctl vm.drop_caches=3 
        LOG_CLEAR_CACHES="from mlperf_logging.mllog import constants as mlperf_constants; from mlperf_log_utils import mx_resnet_print, mlperf_submission_log; mlperf_submission_log(mlperf_constants.RESNET); mx_resnet_print(mlperf_constants.CACHE_CLEAR, val=True, stack_offset=0)"
        python -c "${LOG_CLEAR_CACHES}" 2>&1 |tee -a ${LOGDIR}/${DATESTAMP}_run.log 
	SEED=$(($RANDOM%65535)) timeout 3h mpirun -np 8 --bind-to none --allow-run-as-root run_and_time.sh  2>&1 |tee -a ${LOGDIR}/${DATESTAMP}_run.log 
done

