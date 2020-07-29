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


echo $BASE_DIR
source ml_perf/scripts/common.sh
HOSTNAME=`hostname`
# Sync selfplay periodically
ABORT_FILE=${abort_file} \
ml_perf/scripts/loop.sh rsync -r --append --delete-before ${signal_dir} `dirname ${local_signal_dir}` 2>&1 | tee ${log_dir}/sync_signal_loop_${HOSTNAME}.log > /dev/null &
