#!/bin/bash 
# runs benchmark and reports time to convergence 

CONFIG=${CONFIG:-"mlperf_fp16.json"}
AUC_THRESHOLD=${AUC_THRESHOLD:-0.8025}
BATCH_SIZE=${BATCH_SIZE:-16384}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
LR=${LR:-8}

BIND=''
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
    BIND='./bind.sh --cpu=exclusive --'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
    BIND='./bind.sh --cpu=dgxa100_ccx.sh --mem=dgxa100_ccx.sh'
fi

START_TIMESTAMP=$(date +%s)
huge_ctr --train ${CONFIG} | tee /tmp/dlrm_hugectr.log

cat /tmp/dlrm_hugectr.log | python3 -m mlperf_logger.format_ctr_output ${CONFIG} $START_TIMESTAMP

ret_code=${PIPESTATUS[0]}
set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi


result=`grep -i "hit target" /tmp/dlrm_hugectr.log | awk -F " " '{print $(NF-1)}'`
# exit -1 is accuracy is not hit
if [ -z $result ]; then
  echo "Didn't hit target AUC $AUC_THRESHOLD"
  exit -1
fi


echo "RESULT,DLRM,$result,$USER"
