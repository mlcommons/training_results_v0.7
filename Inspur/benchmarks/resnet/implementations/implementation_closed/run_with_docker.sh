#!/bin/bash
set -euo pipefail

# Vars without defaults
: "${NFSYSTEM:?NFSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/raid/datasets/train-val-recordio-passthrough}"
: "${LOGDIR:=$(pwd)/results}"
: "${COPY_DATASET:=}"

echo $COPY_DATASET

if [ ! -z $COPY_DATASET ]; then
    readonly copy_datadir=$COPY_DATASET
    mkdir -p "${DATADIR}"
    ${CODEDIR}/copy-data.sh "${copy_datadir}" "${DATADIR}"
    ls ${DATADIR}
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _config_file="./config_${NFSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=image_classification
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results")

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    echo "${PRETTY_NAME} / ${NF_PRETTY_NAME:-???} ${NF_OTA_VERSION:-${NF_SWBUILD_VERSION:-???}}"
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
source "${_config_file}"
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
nvidia-docker run --rm --init --detach \
    -v /home/zrg/mlperf/training_v0.7_v2/resnet/scripts:/workspace/image_classification \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        docker exec -it "${_cont_name}" python -c "
import mlperf_log_utils
from mlperf_logging.mllog import constants

mlperf_log_utils.mlperf_submission_log(constants.RESNET)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
import mlperf_log_utils
from mlperf_logging.mllog import constants

mlperf_log_utils.mx_resnet_print_event(key=constants.CACHE_CLEAR, val=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        docker exec -it "${_config_env[@]}" "${_cont_name}" ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
