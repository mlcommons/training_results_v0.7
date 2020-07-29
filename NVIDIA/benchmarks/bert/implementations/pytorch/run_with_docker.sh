#!/bin/bash
set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${LOGDIR:=$(pwd)/results}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=${CONT:-language_model}
_cont_mounts=("--volume=${DATADIR}:/workspace/data" "--volume=${DATADIR_PHASE2}:/workspace/data_phase2" "--volume=${CHECKPOINTDIR}:/results" "--volume=${CHECKPOINTDIR_PHASE1}:/workspace/phase1" "--volume=${EVALDIR}:/workspace/evaldata")

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
source "${_config_file}"
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

PHASE1="\
    --train_batch_size=$BATCHSIZE \
    --learning_rate=${LR:-6e-3} \
    --warmup_proportion=${WARMUP_PROPORTION:-0.0} \
    --max_steps=7038 \
    --num_steps_per_checkpoint=2500 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --input_dir=/workspace/data \
    "
PHASE2="\
    --train_batch_size=$BATCHSIZE \
    --learning_rate=${LR:-4e-3} \
    --opt_lamb_beta_1=${OPT_LAMB_BETA_1:-0.9} \
    --opt_lamb_beta_2=${OPT_LAMB_BETA_2:-0.999} \
    --warmup_proportion=${WARMUP_PROPORTION:-0.0} \
    --max_steps=$MAX_STEPS \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=/workspace/data_phase2 \
    --init_checkpoint=/workspace/phase1/model.ckpt-28252.pt \
    "
PHASES=( "$PHASE1" "$PHASE2" )

PHASE=${PHASE:-2}

cluster=''
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
    cluster='circe'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
    cluster='selene'
fi    

MAX_SAMPLES_TERMINATION=${MAX_SAMPLES_TERMINATION:-14000000}
EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-3000000}
EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-500000}

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' '--cpu=exclusive' '--ib=single' '--cluster=${cluster}' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( 'python' '-u' '-m' 'bind_pyt' "--nsockets_per_node=${DGXNSOCKET}" \
    "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
fi

# Run fixed number of training samples
BERT_CMD="\
    ${CMD[@]} \
    /workspace/bert/run_pretraining.py \
    $PHASE2 \
    --do_train \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=0.712 \
    --max_samples_termination=${MAX_SAMPLES_TERMINATION} \
    --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES} --eval_iter_samples=${EVAL_ITER_SAMPLES} \
    --eval_batch_size=16 --eval_dir=/workspace/evaldata \
    --output_dir=/results \
    --fp16 --fused_gelu_bias --dense_seq_output \
    --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=${GRADIENT_STEPS:-2} \
    --log_freq=1 \
    --bert_config_path=/workspace/phase1/bert_config.json"


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
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        docker exec -it "${_cont_name}" python -c "
import mlperf_logger 
from mlperf_logging.mllog import constants 
mlperf_logger.mlperf_submission_log(\"language_model\")"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
from mlperf_logging.mllog import constants 
from mlperf_logger import log_event 
log_event(key=constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        docker exec -it "${_config_env[@]}" "${_cont_name}" sh -c "./run_and_time.sh \"${BERT_CMD}\" ${SEED:-$RANDOM}"
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
