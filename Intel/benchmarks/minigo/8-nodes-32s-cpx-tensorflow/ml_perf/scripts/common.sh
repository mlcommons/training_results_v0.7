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

# Sets up common state used by other MLPerf scripts.


set -euo pipefail

# Parse the command line arguments, converting each one into a bash variable.
while getopts “-:” opt; do
  case $opt in
    -)
      arg="${OPTARG%=*}"
      val="${OPTARG#*=}"
      eval "${arg}=${val}" ;;
  esac
done

# Assign default values to unset command line arguments.
if [ -z "${board_size-}" ]; then board_size="19"; fi
if [ -z "${flag_dir-}" ]; then flag_dir="${base_dir}/flags"; fi
if [ -z "${golden_chunk_dir-}" ]
  then golden_chunk_dir="${base_dir}/data/golden_chunks"
fi
if [ -z "${golden_chunk_local_dir-}" ]
  then golden_chunk_local_dir="/tmp/golden_chunks"
fi
if [ -z "${golden_chunk_tmp_dir-}" ]
  then golden_chunk_tmp_dir="/tmp/golden_chunks_tmp"
fi
if [ -z "${holdout_dir-}" ]; then holdout_dir="${base_dir}/data/holdout"; fi
if [ -z "${log_dir-}" ]; then log_dir="${base_dir}/logs"; fi
if [ -z "${model_dir-}" ]; then model_dir="${base_dir}/models"; fi
if [ -z "${selfplay_dir-}" ]; then selfplay_dir="${base_dir}/data/selfplay"; fi
if [ -z "${sgf_dir-}" ]; then sgf_dir="${base_dir}/sgf"; fi
if [ -z "${work_dir-}" ]; then work_dir="${base_dir}/work_dir"; fi
if [ -z "${window_size-}" ]; then window_size="5"; fi
if [ -z "${tpu_name-}" ]; then tpu_name=""; fi

# move pause/abort file to separate dir
if [ -z "${signal_dir-}" ]; then signal_dir="${base_dir}/signal"; fi
if [ -z "${abort_file-}" ]; then abort_file="${signal_dir}/abort"; fi
if [ -z "${pause_file-}" ]; then pause_file="${signal_dir}/pause"; fi

# setup a local selfplay dir to allow fast access selfplay games
# through local storage
# note: the last level directory name MUST match the last level
#       directory name of $selfplay_dir
selfplay_local_dir=/tmp/selfplay

# setup a per-node signal dir for selfplay instances to read pause/abort files
local_signal_dir=/tmp/signal

# Preserve the arguments the script was called with.
script_args=("$@")

function clean_dir {
  dir="${1}"
  if [[ "${dir}" == gs://* ]]; then
    # `gsutil rm -f` "helpfully" returns a non-zero error code if the requested
    # target files don't exist.
    set +e
    gsutil -m rm -rf "${dir}"/*
    set -e
  else
    mkdir -p "${dir}"
    rm -rf "${dir}"/*
  fi
}

. ./set_avx2_build

BAZEL_OPTS="-c opt --config=mkl \
            --action_env=PATH \
            --action_env=LD_LIBRARY_PATH \
            $BAZEL_BUILD_OPTS \
            --copt=-DINTEL_MKLDNN"

NUMA_COUNT=`cat /proc/cpuinfo |grep physical\ id|sort -u |wc -l`
VIRT_CORES=`cat /proc/cpuinfo |grep physical\ id|wc -l`
NUMA_CORES=`cat /proc/cpuinfo |grep cpu\ cores|head -n 1|awk '//{print $4}'`
PHY_CORES=$(expr $NUMA_CORES \* $NUMA_COUNT)

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`/lib

set -x
