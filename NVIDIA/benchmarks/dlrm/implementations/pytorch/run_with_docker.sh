#!/bin/bash
#SBATCH --job-name dlrm
#SBATCH -t 02:00:00

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${MOUNTS:='/raid/:/raid/,/gpfs/fs1:/gpfs/fs1'}"
: "${LOGDIR:=./results}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=dlrm
_cont_mounts=("--volume=${DATADIR}:${DATADIR}" "--volume=${LOGDIR}:${LOGDIR}")

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
source "${_config_file}"
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(DATASET_TYPE)
_config_env+=(DGXSYSTEM)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
      docker exec -it "${_cont_name}" python -c "
from dlrm import mlperf_logger
mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)"
    fi
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    docker exec -it ${_config_env[@]} ${_cont_name} python -m torch.distributed.launch --no_python --use_env --nproc_per_node ${DGXNGPU} bash ./run_and_time.sh
  ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
