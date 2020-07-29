#!/bin/bash
#SBATCH -N 1                    # number of nodes
#SBATCH -t 12:00:00             # wall time
#SBATCH -J image_classification # job name
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --threads-per-core=2    # HT is on
#SBATCH --cores-per-socket=24   # 24 cores on each socket 
#SBATCH --overcommit

DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}

## Data, container and volumes
BENCHMARK=${BENCHMARK:-"image_classification"}
BENCHMARK_NAME="RESNET"
CONT=${CONT:-"mlperf-fujitsu:$BENCHMARK"}
DATADIR=${DATADIR:-"/raid/data/imagenet/train-val-recordio-passthrough"} 
LOGDIR=${LOGDIR:-"/raid/results/$BENCHMARK"}
SEED=${SEED:-$(od -A n -t d -N 3 /dev/urandom)} # Allows passing SEED in, which is helpful if NEXP=1 ; for NEXP>1 we need to pick different seeds for subsequent runs
#CODEDIR=${CODEDIR:-$(readlink -f $(dirname ${BASH_SOURCE[0]}))} # SLURM normally copies these files to someplace in /var by default, so best to set this explicitly
#SRCDIR=${SRCDIR:-${CODEDIR}/${BENCHMARK}/mxnet}

# FIXME: having the container running as non-root isn't yet tested
CONTAINER_UID=0
CONTAINER_GID=0

# MLPerf-specific stuff
NEXP=${NEXP:-5} # Default number of times to run the benchmark
CLEAR_CACHES=1
SYSLOGGING=1
LOG_CLEAR_CACHES="'from mlperf_logging.mllog import constants as mlperf_constants; from mlperf_log_utils import mx_resnet_print; mx_resnet_print(mlperf_constants.CACHE_CLEAR, val=True, stack_offset=0)'"
SYS_LOG_GET="'from mlperf_logging.mllog import constants as mlperf_constants; from mlperf_log_utils import mlperf_submission_log; mlperf_submission_log(mlperf_constants.$BENCHMARK_NAME)'"
################################################################################
## DO NOT CHANGE ANYTHING BELOW -- DL params are in run_and_time.sh and config_<system>.sh files 
################################################################################

## Load system-specific parameters for benchmark
PGSYSTEM=${PGSYSTEM:-"PG"}
if [[ ! -f "config_${PGSYSTEM}.sh" ]]; then
  echo "ERROR: Unknown system config ${PGSYSTEM}"
  exit 1
fi
source config_${PGSYSTEM}.sh
export PGSYSTEM
IBDEVICES=${IBDEVICES:-$PGIBDEVICES}

## Check whether we are running in a slurm env
INSLURM=1
if [[ -z "$SLURM_JOB_ID" ]]; then
  INSLURM=0
  export SLURM_JOB_ID="${DATESTAMP}"
  export SLURM_JOB_NUM_NODES=1
  export OMPI_COMM_WORLD_LOCAL_RANK=0
fi
if [[ -z "$SLURM_NTASKS_PER_NODE" ]]; then
  export SLURM_NTASKS_PER_NODE="${PGNGPU}"
fi

# Create results directory
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -m 777 -p $(dirname "${LOGFILE_BASE}")

## Docker volume setup
export VOLS="-v $DATADIR:/data -v $LOGDIR:/results"
#for i in $(ls run.sub run_and_time.sh config_*.sh ompi_bind_*.sh *.py); do
#  VOLS+=" -v $SRCDIR/$i:/workspace/$BENCHMARK/$i"
#done

export MLPERF_HOST_OS="Ubuntu 18.04"

# Preset env vars to pass through 'docker run', 'docker exec', and 'mpirun'
VARS=(
       -e "CONT=${CONT}"
       -e "SEED"
       -e "MLPERF_HOST_OS"
       -e "PGSYSTEM=${PGSYSTEM}"
       -e "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES}"
       -e "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}"
       -e "OMPI_MCA_mca_base_param_files=/dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf"
     )

## Set container name to be used on all nodes
export CONTNAME="mpi_${SLURM_JOB_ID}"

## Check whether we're in a multi-node configuration
if [[ $SLURM_JOB_NUM_NODES -eq 1 ]]; then
  hosts=( `hostname` )
  IBDEVICES="" # don't need IB if not multi-node
  SRUN=""
  SRUNl=""
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
  SRUN="srun --mem=0 -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1"
  SRUNl="$SRUN -l"
fi
if [[ ! $KVSTORE =~ "horovod" ]]; then
  MPIRUN=( )
else
  [[ "$DEBUG" != "0" ]]
  MPIRUN=( mpirun --allow-run-as-root --bind-to none ${VARS[@]//-e/-x} --launch-agent "docker exec $CONTNAME orted" )
fi

## Pull image on all nodes by default
[[ "${PULL}" -ne "0" ]] && $SRUNl docker pull $CONT

##########################################
## Configure multinode
##########################################
if [[ $SLURM_JOB_NUM_NODES -gt 1 || $KVSTORE =~ "horovod" ]]; then

  # systemd needs to be told not to wipe /dev/shm when the user context ends.
  #$SRUNl grep -iR RemoveIPC /etc/systemd

  # 1. Prepare run dir on all nodes
  mkdir -p /dev/shm/mpi/${SLURM_JOB_ID}; chmod 700 /dev/shm/mpi/${SLURM_JOB_ID}

  # 2. Create mpi hostlist
  rm -f /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
  for hostn in ${hosts[@]}; do
     echo "$hostn slots=${SLURM_NTASKS_PER_NODE}" >> /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
  done

  # 3. Create mpi config file
  cat > /dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf <<EOF
plm_rsh_agent = /usr/bin/ssh
plm_rsh_args = -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -oBatchMode=yes -l ${USER}
orte_default_hostfile = /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
btl_openib_warn_default_gid_prefix = 0
mpi_warn_on_fork = 0
EOF

  # 4. Distribute config file, host list to all nodes
  tar zcPf /dev/shm/mpi/${SLURM_JOB_ID}.tgz /dev/shm/mpi/${SLURM_JOB_ID}/
  cat /dev/shm/mpi/${SLURM_JOB_ID}.tgz | $SRUNl tar zxPf -
  rm -f /dev/shm/mpi/${SLURM_JOB_ID}.tgz

  # 5. Grab SSH keys and configs on each node
  $SRUNl cp -pr ~/.ssh /dev/shm/mpi/${SLURM_JOB_ID}/
fi

##########################################
## Start up container on each node
##########################################
DOCKERRUN_ARGS=(
                 --rm
                 --net=host
                 --uts=host
                 --ipc=host
                 --ulimit stack=67108864
                 --ulimit memlock=-1
                 --security-opt seccomp=unconfined
                 -u $CONTAINER_UID:$CONTAINER_GID
                 $IBDEVICES
               )

echo
echo "Creating containers:"
$SRUNl bash -c "echo -n 'Launching on node ' && hostname"
$SRUNl nvidia-docker run -v $PWD:/workspace/image_classification --init -d "${DOCKERRUN_ARGS[@]}" --name $CONTNAME $VOLS "${VARS[@]}" $CONT bash -c 'rm -f /etc/shinit && sleep infinity' ; rv=$?

[[ $rv -ne 0 ]] && echo "ERR: Container launch failed." && exit $rv

sleep 30

docker logs $CONTNAME

if [[ $SLURM_JOB_NUM_NODES -gt 1 ]] || [[ $PGSYSTEM =~ "_multi_" ]]; then
  # For multinode, each container needs to be able to have access to the host system user's SSH keys & config
  # FIXME: this probably doesn't work unless the UID/GID matches (untested) and/or the container is running as root (default)
  $SRUNl docker exec "${VARS[@]}" $CONTNAME bash -c "cp -pr /dev/shm/mpi/${SLURM_JOB_ID}/.ssh ~/ ; chown -R $CONTAINER_UID:$CONTAINER_GID ~/.ssh/ ; chmod 700 ~/.ssh/"
fi

##########################################
## Launch app into running container(s)
##########################################

export SEED

rvs=0
for nrun in `seq 1 $NEXP`; do
  (
    echo "Beginning trial $nrun of $NEXP"

    echo "Run vars: id $SLURM_JOB_ID"

    if [[ $SYSLOGGING -eq 1 ]]; then 
      # Clear RAM cache dentries and inodes
        VARS_STR="${VARS[@]}"
        bash -c "echo -n 'Gathering sys log on ' && hostname && docker exec $VARS_STR $CONTNAME python -c ${SYS_LOG_GET}"
        if [[ $? -ne 0 ]]; then
            echo "ERR: Sys log gathering failed."
            exit 1
        fi
    fi

    if [[ $CLEAR_CACHES -eq 1 ]]; then
      # Clear RAM cache dentries and inodes
      if [[ $INSLURM -eq 1 ]]; then
        VARS_STR="${VARS[@]}"
        $SRUN bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3 && docker exec $VARS_STR $CONTNAME python -c ${LOG_CLEAR_CACHES}" \
          || echo "ERR: Cache clearing failed."
      else
        echo "Clearing cache on $(hostname)"
        docker run --init --rm --privileged --entrypoint bash "${VARS[@]}" $CONT -c "sync && echo 3 > /proc/sys/vm/drop_caches && python -c ${LOG_CLEAR_CACHES} || exit 1" \
          || echo "ERR: Cache clearing failed."
      fi
    fi

    # Launching app
    echo 
    echo "Launching user script on master node:"
    docker exec "${VARS[@]}" $CONTNAME "${MPIRUN[@]}" ./run_and_time.sh ; exit $?

  ) |& tee ${LOGFILE_BASE}_$nrun.log

  rv=${PIPESTATUS[0]}
  [[ $rv -ne 0 ]] && echo "ERR: User script failed (nrun=$nrun; rv=$rv)." && let rvs+=$rv

  ## SEED update
  export SEED=$(od -A n -t d -N 3 /dev/urandom)

done

##########################################
## Clean up
##########################################

# Shut down containers and clean up /dev/shm
$SRUNl timeout -k 20 10 docker rm -f $CONTNAME >/dev/null
$SRUNl rm -rf /dev/shm/mpi/${SLURM_JOB_ID}/

# Finish
[[ $rvs -ne 0 ]] && echo "ERR: User script failed." && exit $rvs
exit 0
