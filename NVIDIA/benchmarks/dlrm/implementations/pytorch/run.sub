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
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=dlrm

# Setup container
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    if [[ $CLEAR_CACHES == 1 ]]; then
      srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
      srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
from dlrm import mlperf_logger
mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)"
    fi
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    srun --mpi=none --ntasks="${DGXNGPU}" --ntasks-per-node="${DGXNGPU}" \
         --container-image="${CONT}" --container-mounts="${MOUNTS}" \
         /bin/bash ./run_and_time.sh
  ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
