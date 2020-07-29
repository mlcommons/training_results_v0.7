#!/bin/bash

model_dir="/data/resnet50"
data_dir="/data/imagenet_tf"
log_dir="/data/tf2"
repo_dir="staging_clean/models/dev"
docker_image="sgpyc/mlperf_v0.7:resnet_20200620"
host_workspace=$HOME
host_data_dir="/data"

env_params=""
env_params+="-e PYTHONPATH=/workspace/${repo_dir} "
env_params+="-e CUDA_VISIBLE_DEVICES=0,1,3,2,7,6,4,5  "

runner="python3"
target="/workspace/${repo_dir}/tf2_resnet/resnet_ctl_imagenet_main.py"

num_runs=5
if [ ! -z $NUM_RUNS ]; then
  num_runs=$NUM_RUNS
fi

num_gpus=8
per_gpu_batch_size=312
batch_size=$(($per_gpu_batch_size * $num_gpus))
steps_per_loop=$((1281167 / $batch_size + 1))

# commandline flags to the training script
params=""
params+="--num_gpus=$num_gpus "
params+="--dtype=fp16 "
params+="--batch_size=$batch_size "
params+="--base_learning_rate=9.5 "
params+="--train_epochs=40 "
params+="--warmup_epochs=5 "
params+="--eval_offset_epochs=3 "

params+="--epochs_between_evals=4 "
params+="--optimizer=LARS "
params+="--lr_schedule=polynomial "
params+="--label_smoothing=0.1 "
params+="--weight_decay=0.0002 "

params+="--steps_per_loop=$steps_per_loop "
params+="--enable_eager=true "
params+="--report_accuracy_metrics=false "
params+="--log_steps=125 "
params+="--tf_gpu_thread_mode=gpu_private "
params+="--datasets_num_private_threads=32 "
params+="--training_dataset_cache=true "
params+="--data_dir=$data_dir "
params+="--model_dir=$model_dir "
params+="--clean "
params+="--single_l2_loss_op "
params+="--training_prefetch_batchs=128 "
params+="--eval_dataset_cache=true "
params+="--eval_prefetch_batchs=192 "
params+="--enable_device_warmup=true "
params+="--verbosity=0 "

docker_flags=""
docker_flags+="--runtime=nvidia "
docker_flags+="--rm "
docker_flags+="--net=host "
docker_flags+="--uts=host "
docker_flags+="--ipc=host "
docker_flags+="--ulimit stack=67108864 "
docker_flags+="--ulimit memlock=01 "
docker_flags+="--security-opt seccomp=unconfined "
docker_flags+="--privileged=true "
docker_flags+="-v $host_data_dir:/data "
docker_flags+="-v $host_workspace:/workspace "
docker_flags+="--name=tf2-resnet "

docker_exec="sudo docker exec ${env_params[@]} tf2-resnet"

echo "Creating docker container tf2-resnet from image ${docker_image} ..."
cmd="sudo docker run ${docker_flags[@]} ${docker_image} sleep infinity"
echo "${cmd[@]} &"
      ${cmd[@]} &
sleep 20

for i in $(seq 1 1 $num_runs); do
  echo "======= Run $(($i - 1))  of $num_runs ========="
  echo "Clearing host cache ..."
  cmd="sync; echo 3 > /proc/sys/vm/drop_caches"
  bash -c '${cmd[@]}'

  echo "Clearing container cache ..."
  ${docker_exec[@]} bash -c '${cmd[@]}'

  timestamp=$(date "+%Y%m%d%H%M%S")
  cmd="${runner[@]} ${target[@]} ${params[@]}"
  log="resnet50_$(($i - 1))_of_${num_runs}_$timestamp.txt"
  echo "Training model and piping output to ${host_workspace}/${repo_dir}/$log ..."
  echo "${docker_exec[@]} bash -c \"${cmd[@]} > /workspace/${repo_dir}/$log 2>&1\""
        ${docker_exec[@]} bash -c "${cmd[@]} > /workspace/${repo_dir}/$log 2>&1"

  sleep 1
  echo ""
done

echo "Stopping docker container tf2-resnet ..."
cmd="sudo docker container stop tf2-resnet"
echo ${cmd[@]}
     ${cmd[@]}
