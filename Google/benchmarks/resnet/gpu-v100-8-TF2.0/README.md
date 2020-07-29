# ResNet

## Benchmark Information

ResNet is the
[ResNet-50 image classification](https://github.com/mlperf/training/tree/master/image_classification) benchmark.

## Software

[Tensorflow 2.0](https://www.tensorflow.org/)

## Hardware

Both GPU-V100-8 and TPU-V3-32.

## Model
### Publication/Attribution

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. [Deep Residual Learning
for Image Recognition](https://arxiv.org/abs/1512.03385). CVPR 2016.

## Dataset Preparation

*   [ImageNet dataset preparation](https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment)


## Directions

### On GPU-V100-8

- Create a n1-highmem-96 GCP compute instance with 8 V100 GPUs and 8 local
NVME SSDs:

  ```
  gcloud compute instances create <instance name> \
      --accelerator type=nvidia-tesla-v100,count=8 \
      --boot-disk-auto-delete \
      --boot-disk-size 100 \
      --boot-disk-type pd-ssd \
      --format json \
      --image-family common-cu101 \
      --image-project deeplearning-platform-release \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --local-ssd interface=NVME \
      --machine-type n1-highmem-96 \
      --maintenance-policy TERMINATE \
      --metadata timeout_utc=install-nvidia-driver=True \
      --network default \
      --network-tier PREMIUM \
      --no-restart-on-failure \
      --project <GCP project id> \
      --quiet \
      --zone <GCP zone>
  ```

  Form a RAID-0 disk array consists of the 8 SSDs, and mount to `/data`.

- Prepare the host enviroment with the following:

  - NVIDIA Driver that supports at least CUDA 10.1 for V100 GPUs
  - Docker CE
  - NVIDIA Docker

- Prepare the docker image

  A [docker image](https://hub.docker.com/layers/sgpyc/mlperf_v0.7/resnet_20200620/images/sha256-099061e9676643798567c9915ad017507ea3d9ab563ce6031817e1ff69470905?context=repo) based on [tensorflow/tensorflow:nightly-gpu](https://hub.docker.com/layers/tensorflow/tensorflow/nightly-gpu/images/sha256-d0472ab78c4184ba9e7134d753906d7297b141cbc265133bd1aaee401c09e271?context=explore)  and pointing to the nightly build that this submition has been tested with, can be either:

  Build from scratch:

  ```
  sudo docker build -t sgpyc/mlperf_v0.7:resnet_20200620 ./
  ```

  Or, pull from Docker hub:

  ```
  sudo docker pull sgpyc/mlperf_v0.7:resnet_20200620
  ```

- Download the ImageNet dataset to `/data/imagenet_tf`

- Run the benchmark

  ```
  ./run_and_time.sh
  ```

  Before running, it may require to modify the dirtory paths in the first few lines, to ponit to the correct directories.

### On TPU-V3-32

The TPU run uses the [PerfZero](https://github.com/tensorflow/benchmarks/tree/master/perfzero) benchmarking inferstructure.

- Create a n1-standard-96 GCP compute instance, and attach V3-32 TPUs.

- Build a docker image from the PerfZero project:

  ```
  git clone https://github.com/tensorflow/benchmarks.git
  cd benchmarks
  git checkout 449e900ef018a775f2827dd3e591900c761004ab
  cd perfzero/docker
  mkdir empty_dir
  touch empty_dir/EMPTY

  sudo docker build \
    --no-cache \
    --pull \
    -t perfzero/tensorflow \
    --build-arg tensorflow_pip_spec=tf-nightly-gpu==2.3.0.dev20200620 \
    --build-arg local_tensorflow_pip_spec=EMPTY \
    --build-arg extra_pip_specs='' \
    -f ./Dockerfile_ubuntu_1804_tf_v2_1 \
    ./empty_dir
  ```

- Run the benchmark

  ```
  sudo docker run --rm \
  -v $HOME:/workspace \
  -v /data:/data \
  perfzero/tensorflow python3 \
  /workspace/benchmarks/perfzero/lib/benchmark.py \
  --bigquery_dataset_table_name="" \
  --ml_framework_build_label=nightly-mlperf-tpu \
  --execution_label=prod \
  --platform_name=kokoro-gcp \
  --system_name=n1-standard-96 \
  --output_gcs_url= \
  --benchmark_num_trials=1 \
  --scratch_gcs_url=<some GCS bucket> \
  --root_data_dir=/data \
  --bigquery_project_name= \
  '--git_repos=<mlperf_repo>;<branch>;<commit hash>' \
  --data_downloads= \
  --python_path=<path to where tf2_resnet stays> \
  --benchmark_methods=tf2_resnet.resnet_benchmark.ResnetBenchmarkReal.benchmark_4x4_tpu \
  --gcloud_key_file_url=<some keyfile> \
  --result_upload_methods=perfzero_export_lib.upload_summary '--tpu_parameters={\
      "name": "somename", \
      "project": "someproject", \
      "zone": "somezone", \
      "size": "v3-32", \
      "version": "2.3.0.dev20200620"}'
  ```

  The actual flags passed to the training script are:

  ```
  --base_learning_rate=14.0 \
  --batch_size=4096 \
  --cache_decoded_image \
  --clean \
  --data_dir=gs://mlperf-imagenet/imagenet/combined \
  --distribution_strategy=tpu \
  --dtype=bf16 \
  --enable_device_warmup \
  --enable_eager \
  --epochs_between_evals=4 \
  --eval_dataset_cache \
  --eval_offset_epochs=0 \
  --label_smoothing=0.1 \
  --log_steps=125 \
  --lr_schedule=polynomial \
  --model_dir=<some GCS bucket/some dir> \
  --optimizer=LARS \
  --noreport_accuracy_metrics \
  --single_l2_loss_op \
  --steps_per_loop=313 \
  --tpu=<some TPU> \
  --tpu_zone=<some TPU zone> \
  --train_epochs=41 \
  --training_dataset_cache \
  --verbosity=0 \
  --warmup_epochs=2 \
  --weight_decay=0.0002
  ```
  
## Submission CLs

Framework	Platform	Benchmark	TPU Worker CL	TPU Green VM	FLAX Commit	JAX Commit
TF	tpu-v4-512-TF	gnmt	317991884	n/a	n/a	n/a
TF	tpu-v4-512-TF	transformer	317773961	n/a	n/a	n/a
TF	tpu-v4-512-TF	bert	318324155	n/a	n/a	n/a
TF	tpu-v4-512-TF	resnet	316295790	n/a	n/a	n/a
TF	tpu-v4-512-TF	maskrcnn	317770796	n/a	n/a	n/a
TF	tpu-v4-512-TF	ssd	316295790	n/a	n/a	n/a
TF	tpu-v4-128-TF	minigo	316295790	n/a	n/a	n/a
TF	tpu-v4-128-TF	gnmt	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	transformer	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	bert	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	resnet	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	maskrcnn	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	dlrm	317896100	n/a	n/a	n/a
TF	tpu-v4-128-TF	ssd	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	gnmt	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	transformer	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	bert	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	resnet	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	maskrcnn	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	dlrm	317896100	n/a	n/a	n/a
TF	tpu-v4-16-TF	ssd	317896100	n/a	n/a	n/a
TF	tpu-v3-1024-TF	gnmt	317991884	n/a	n/a	n/a
TF	tpu-v3-1024-TF	maskrcnn	317770796	n/a	n/a	n/a
TF	tpu-v3-8192-TF	transformer	317896100	n/a	n/a	n/a
TF	tpu-v3-8192-TF	bert	317585047	n/a	n/a	n/a
TF	tpu-v3-8192-TF	resnet	314149382	n/a	n/a	n/a
TF	tpu-v3-8192-TF	ssd	314149382	n/a	n/a	n/a
TF2	tpu-v3-32-TF2.0	bert	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
TF2	tpu-v3-32-TF2.0	resnet	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
TF2	gpu-v100-8-TF2.0	bert	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
TF2	gpu-v100-8-TF2.0	resnet	2.3.0-dev20200620	2.3.0-dev20200620	n/a	n/a
JAX	tpu-v3-8192-JAX	transformer	cl/318350991	n/a	https://github.com/google/jax/commit/c9670d50c5015f36d7569bdf204b0651c64e55fb	https://github.com/google/flax/commit/ee20ca22dda285dac41e3e65f0a2d4cedc88bea8 
JAX	tpu-v3-8192-JAX	bert	cl/318008749	n/a	https://github.com/google/jax/commit/75278309aa00cd22e6b85422215383fa9d594ecb	https://github.com/google/flax/commit/ed42d0670c293d0ac205313b19fbef9832e81304 
JAX	tpu-v3-8192-JAX	resnet	cl/316292965	n/a	https://github.com/google/flax/commit/484cc11669f773ba69d184f2f26933954797943e	https://github.com/google/flax/commit/484cc11669f773ba69d184f2f26933954797943e
JAX	tpu-v3-4096-JAX	ssd	cl/317422581	n/a	https://github.com/google/jax/commit/8f4ba7e679b889ce8b75ef8fa07a1df947d89e52	https://github.com/google/flax/commit/4578df437833933fb5e24ac732d0cf7a5ed7d0e7 


