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

# Bootstraps a reinforcement learning loop from a checkpoint generated from
# a previous run.
#
# Example usage:
#  ./ml_perf/scripts/init_from_checkpoint.sh \
#    --board_size=19 \
#    --base_dir="${BASE_DIR}" \
#    --checkpoint_dir="${SOURCE_CHECKPOINT_DIR}"


source ml_perf/scripts/common.sh


# Build the C++ binaries
bazel build $BAZEL_OPTS \
  --copt=-O3 \
  --define=board_size="${board_size}" \
  --define=tf=1 \
  cc:concurrent_selfplay cc:sample_records cc:eval

# Initialize a clean directory structure.
for var_name in flag_dir golden_chunk_dir golden_chunk_tmp_dir holdout_dir \
                log_dir model_dir selfplay_dir sgf_dir work_dir signal_dir; do
  clean_dir "${!var_name}"
done
rm -f "${abort_file}"

if [[ -v HOSTLIST ]]
then
  echo Create golden chunk dir for each node
  mpirun -bootstrap ssh -ppn 1 -hosts $HOSTLIST mkdir -p $golden_chunk_local_dir
  mpirun -bootstrap ssh -ppn 1 -hosts $HOSTLIST rm -f $golden_chunk_local_dir/*
  echo Create signal dir for each node
  mpirun -bootstrap ssh -ppn 1 -hosts $HOSTLIST mkdir -p $local_signal_dir
  mpirun -bootstrap ssh -ppn 1 -hosts $HOSTLIST rm -f $local_signal_dir/*
fi

BOARD_SIZE="${board_size}" \
python3 ml_perf/init_from_checkpoint.py \
  --checkpoint_dir="${checkpoint_dir}" \
  --selfplay_dir="${selfplay_dir}" \
  --work_dir="${work_dir}" \
  --flag_dir="${flag_dir}" \
  --board_size="${board_size}"

# copy selfplay files to local storage
echo "Copy selfplay files to local storages"
mkdir -p ${selfplay_local_dir}
rm -rf ${selfplay_local_dir}/*
#cp -r ${selfplay_dir}/* ${selfplay_local_dir}
rsync -r --append ${selfplay_dir} `dirname ${selfplay_local_dir}`
ls ${selfplay_local_dir}
