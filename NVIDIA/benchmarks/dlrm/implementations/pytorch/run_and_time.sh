#!/bin/bash 
# runs benchmark and reports time to convergence 


AUC_THRESHOLD=${AUC_THRESHOLD:-0.8025}
BATCH_SIZE=${BATCH_SIZE:-16384}
LR_ARGS=${LR_ARGS:-"--lr 8"}
SEED=${SEED:-""}
TEST_AFTER=${TEST_AFTER:-0}
DATASET_TYPE=${DATASET_TYPE:-"dist"}

BIND=''
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
  BIND='./bind.sh --cpu=exclusive --'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
  BIND='./bind.sh --cpu=dgxa100_ccx.sh --mem=dgxa100_ccx.sh'
fi

CACHE_EVAL_DATA="--nocache_eval_data"
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
  CACHE_EVAL_DATA="--cache_eval_data"
fi

seed_arg=""
if [ ! -z "$SEED" ]; then
  seed_arg="--seed ${SEED}";
fi

${BIND} python -u scripts/dist_train.py --model_config dlrm/config/mlperf_40m.limit.json --dataset $DATADIR \
    --batch_size $BATCH_SIZE $LR_ARGS --auc_threshold $AUC_THRESHOLD --test_freq $((209715200 / BATCH_SIZE)) \
    $seed_arg --test_after $TEST_AFTER 2>&1 --dataset_type $DATASET_TYPE $CACHE_EVAL_DATA | tee /tmp/dlrm_${SEED}.log
ret_code=${PIPESTATUS[0]}

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi


result=`grep -i "hit target" /tmp/dlrm_${SEED}.log | awk -F " " '{print $(NF-1)}'`
# exit -1 is accuracy is not hit
if [ -z $result ]; then
  echo "Didn't hit target AUC $AUC_THRESHOLD"
  exit -1
fi


echo "RESULT,DLRM,$SEED,$result,$USER"
