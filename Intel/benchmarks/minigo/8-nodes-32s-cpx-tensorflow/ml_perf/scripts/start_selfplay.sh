# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Starts selfplay processes running.
#
# Example usage that starts 8 processes running on 8 GPUs:
#  ./ml_perf/scripts/start_selfplay.sh \
#    --board_size=19 \
#    --devices=0,1,2,3,4,5,6,7 \
#    --base_dir="${BASE_DIR}"


source ml_perf/scripts/common.sh

NUMA_COUNT=`cat /proc/cpuinfo |grep physical\ id|sort -u |wc -l`
VIRT_CORES=`cat /proc/cpuinfo |grep physical\ id|wc -l`
NUMA_CORES=`cat /proc/cpuinfo |grep cpu\ cores|head -n 1|awk '//{print $4}'`
PHY_CORES=$(expr $NUMA_CORES \* $NUMA_COUNT)

echo "NUMA count = $NUMA_COUNT"
echo "Virtual cores = $VIRT_CORES"
echo "Cores per NUMA = $NUMA_CORES"
echo "Physical cores = $PHY_CORES"

log_dir="${base_dir}/logs/selfplay/`hostname`"
mkdir -p "${log_dir}"

# start syncing pause file written by train_loop to shared file system with local dir
./ml_perf/scripts/start_sync_signal.sh  --base_dir=${base_dir} 2>&1 >/dev/null &

# Run selfplay workers.
for ((device=0;device<PHY_CORES;device++)); do
  OMP_NUM_THREADS=1 \
  KMP_AFFINITY=granularity=fine,proclist=[${device}],explicit \
  ABORT_FILE=${abort_file} \
  ml_perf/scripts/loop.sh numactl --physcpubind=${device} ./bazel-bin/cc/concurrent_selfplay \
    --device="cpu" \
    --flagfile="${flag_dir}/selfplay.flags" \
    --output_dir="${selfplay_dir}/\$MODEL/${device}" \
    --holdout_dir="${holdout_dir}/\$MODEL/${device}" \
    --model="${model_dir}/%d.minigo" \
    --run_forever=1 \
    --abort_file="${local_signal_dir}/abort" \
    --pause_file="${local_signal_dir}/pause" \
    2>&1 | tee "${log_dir}/`hostname`_selfplay_${device}.log" > /dev/null &
done
