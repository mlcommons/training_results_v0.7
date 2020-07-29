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

# Starts the training loop.
# Example usage:
#  ./ml_perf/scripts/train.sh \
#    --board_size=19 \
#    --base_dir="${OUTPUT_DIR}"


source ml_perf/scripts/common.sh


# Set up an exit handler that stops the selfplay workers.
function stop_selfplay {
  ./ml_perf/scripts/stop_selfplay.sh "${script_args[@]}"
}
trap stop_selfplay EXIT

if [[ ! -n "$precision" || $precision != "fp32" ]]; then
  precision=int8
fi

# Run the training loop.
if [[ -v HOSTLIST ]]; then
  echo "Run multi-node training."
  if [[ -v TRAINHOSTLIST ]]; then
    echo Seperate hosts for training
    train_host_list=$TRAINHOSTLIST
  else
    train_host_list=$HOSTLIST
  fi
  echo "Train host list = $train_host_list"
  echo "Physical cores = $PHY_CORES"
  echo "Virtual cores = $VIRT_CORES"
  echo "Cores per NUMA = $NUMA_CORES"
  BOARD_SIZE="${board_size}" \
  CUDA_VISIBLE_DEVICES="0" \
  python3 ml_perf/train_loop.py \
    --flags_dir="${flag_dir}" \
    --golden_chunk_dir="${golden_chunk_dir}" \
    --golden_chunk_local_dir="${golden_chunk_local_dir}" \
    --golden_chunk_tmp_dir="${golden_chunk_tmp_dir}" \
    --holdout_dir="${holdout_dir}" \
    --log_dir="${log_dir}" \
    --model_dir="${model_dir}" \
    --selfplay_dir="${selfplay_local_dir}" \
    --pause="${pause_file}" \
    --work_dir="${work_dir}" \
    --log_dir="${log_dir}" \
    --flagfile="${flag_dir}/train_loop.flags" \
    --window_size="${window_size}" \
    --precision="${precision}" \
    --tpu_name="${tpu_name}" \
    --hostlist="${train_host_list}" \
    --physical_cores=$PHY_CORES \
    --virtual_cores=$VIRT_CORES \
    --numa_cores=$NUMA_CORES \
    2>&1 | tee "${log_dir}/train_loop.log"
else
  echo "Run single node training."
  BOARD_SIZE="${board_size}" \
  CUDA_VISIBLE_DEVICES="0" \
  echo "Physical cores = $PHY_CORES"
  echo "Virtual cores = $VIRT_CORES"
  echo "Cores per NUMA = $NUMA_CORES"
  python3 ml_perf/train_loop.py \
    --flags_dir="${flag_dir}" \
    --golden_chunk_dir="${golden_chunk_dir}" \
    --golden_chunk_local_dir="${golden_chunk_local_dir}" \
    --golden_chunk_tmp_dir="${golden_chunk_tmp_dir}" \
    --holdout_dir="${holdout_dir}" \
    --log_dir="${log_dir}" \
    --model_dir="${model_dir}" \
    --selfplay_dir="${selfplay_local_dir}" \
    --pause="${pause_file}" \
    --work_dir="${work_dir}" \
    --flagfile="${flag_dir}/train_loop.flags" \
    --window_size="${window_size}" \
    --precision="${precision}" \
    --tpu_name="${tpu_name}" \
    --physical_cores=$PHY_CORES \
    --virtual_cores=$VIRT_CORES \
    --numa_cores=$NUMA_CORES \
    2>&1 | tee "${log_dir}/train_loop.log"
fi
