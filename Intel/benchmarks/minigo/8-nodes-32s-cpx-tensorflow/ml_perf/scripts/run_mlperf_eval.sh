# Set the benchmark output base directory.
BASE_DIR=$HOME/mlperf/results-$1

NUMA_COUNT=`cat /proc/cpuinfo |grep physical\ id|sort -u |wc -l`
VIRT_CORES=`cat /proc/cpuinfo |grep physical\ id|wc -l`
NUMA_CORES=`cat /proc/cpuinfo |grep cpu\ cores|head -n 1|awk '//{print $4}'`
PHY_CORES=$(expr $NUMA_CORES \* $NUMA_COUNT)
LAST_PHY_CORE=$(expr $PHY_CORES - 1)

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`/lib

echo "NUMA count = $NUMA_COUNT"
echo "Virtual cores = $VIRT_CORES"
echo "Cores per NUMA = $NUMA_CORES"
echo "Physical cores = $PHY_CORES"
echo "Last physical core = $LAST_PHY_CORE"

# Once the training loop has finished, run model evaluation to find the
# first trained model that's better than the target.
# TODO(tommadams): we still need to do more testing before choosing a
# target model.
python3 ml_perf/eval_models.py \
     --start=50 \
     --flags_dir=ml_perf/flags/$1 \
     --model_dir=$BASE_DIR/models/ \
     --target=ml_perf/target/target.minigo \
     --devices=`seq -s, 0 $LAST_PHY_CORE`
