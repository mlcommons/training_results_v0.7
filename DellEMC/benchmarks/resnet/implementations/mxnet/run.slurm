#!/bin/bash
#SBATCH -N 1                     # number of nodes
#SBATCH -n 4                     # total number of processes
#SBATCH --ntasks-per-node 4      # tasks per node
#SBATCH -t 12:00:00             # wall time
#SBATCH --job-name image_classification
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH -p dedicateq
#SBATCH --gres=gpu:4

set -euxo pipefail

# Vars without defaults
: "${SYSTEM:?SYSTEM not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/mnt/isilon/DeepLearning/database/mlperf/ilsvrc12_passthrough}"
: "${LOGDIR:=./results}"

NUM_NODES=$SLURM_JOB_NUM_NODES
CONT="sandbox-mxnet-ngc20.06"

# Other vars
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"

# MLPerf vars
MLPERF_HOST_OS=$(bash <<EOF
    source /etc/os-release
    echo "\${PRETTY_NAME}"
EOF
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

module load shared extra/openmpi/3.1.6


# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
   
        hosts=( `scontrol show hostname |tr "\n" " "` ) 
        for node_id in `seq 0 $(($NUM_NODES-1))`; do
            # print system info
            srun -N 1 -n 1 -w ${hosts[$node_id]} mpirun --allow-run-as-root -np 1 singularity exec $CONT python -c \
                "import mlperf_log_utils; \
                 from mlperf_logging.mllog import constants; \
                 mlperf_log_utils.mlperf_submission_log(constants.RESNET)"
            
            echo "Clearning cache on each node"
            srun -N 1 -n 1 -w ${hosts[$node_id]} bash -c "sync && echo 3 | tee /proc/sys/vm/drop_caches"
            srun -N 1 -n 1 -w ${hosts[$node_id]} mpirun --allow-run-as-root -np 1 singularity exec $CONT python -c \
                "from mlperf_logging.mllog import constants; \
                 from mlperf_log_utils import mx_resnet_print_event; \
                 mx_resnet_print_event(key=constants.CACHE_CLEAR, val=True)"
        done

	
        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
	    MPIRUN="mpirun --allow-run-as-root --bind-to none -np $SLURM_NTASKS"
	    ${MPIRUN} singularity exec -B /cm/local/apps/cuda/libs/current:/mnt/driver \
                              -B $DATADIR:/data -B $PWD:/mnt/current \
                              $CONT bash /mnt/current/run_and_time_multi.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
