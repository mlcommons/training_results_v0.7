# Bert

## Benchmark Information

Bert is the
[Bert](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert) benchmark.

## Software

[Tensorflow 2.0](https://www.tensorflow.org/)

## Hardware

Both GPU-V100-8 and TPU-V3-32.

## Model
### Publication/Attribution

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers 
for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL-HLT 2019.

## Location of the input files
*   [Input Data Location](https://pantheon.corp.google.com/storage/browser/bert-data-europe/tfrecords3_500_parts)
*   [TF2 Converted Checkpoint Location](https://pantheon.corp.google.com/storage/browser/nnigania_perf_profiles/bert_mlperf_data)

## Dataset Preparation

*   [Bert dataset preparation](https://github.com/mlperf/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets)


## Directions

The runs use the [PerfZero](https://github.com/tensorflow/benchmarks/tree/master/perfzero) benchmarking infrastructure.

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
  sudo nvidia-docker run --rm \
  -v /home/kbuilder:/workspace \
  -v /data:/data \
  perfzero/tensorflow python3 \
  /workspace/benchmarks/perfzero/lib/benchmark.py \
  --bigquery_dataset_table_name="" \
  --ml_framework_build_label=nightly-mlperf-gpu \
  --execution_label=prod \
  --platform_name=kokoro-gcp \
  --system_name=n1-highmem-96-8xV100 \
  --output_gcs_url= \
  --benchmark_num_trials=1 \
  --scratch_gcs_url= \
  --root_data_dir=/data \
  --bigquery_project_name= \
  '--git_repos=<mlperf_repo>;<branch>;<commit hash>' \
  --data_downloads=gs://nnigania_perf_profiles/bert_mlperf_data \
  --python_path=staging/models/prod,mlcompass \
  --benchmark_methods=tf2_bert.bert_benchmark.BertClassifyBenchmarkReal.benchmark_8_gpu \
  --gcloud_key_file_url=gs://tf-performance/auth_tokens/benchmark_upload_gce.json  \
  --result_upload_methods=perfzero_export_lib.upload_summary
  ```

  The actual flags passed to the training script are:

  ```
  --all_reduce_alg=nccl \
  --bert_config_file=/data/bert_mlperf_data/bert_config.json \
  --beta_1=0.91063 \
  --beta_2=0.96497 \
  --device_warmup \
  --do_eval \
  --dtype=fp16 \
  --eval_batch_size=48 \
  --init_checkpoint=/data/bert_mlperf_data/model.ckpt-28252 \
  '--input_files=gs://bert-data-europe/tfrecords3_500_parts/part-*' \
  --learning_rate=0.00035221 \
  --loss_scale=dynamic \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --model_dir=/workspace/output/2020-06-25-18-15-51-461988/benchmark_8_gpu \
  --num_accumulation_steps=19 \
  --num_gpus=8 \
  --num_steps_per_epoch=8000 \
  --num_train_epochs=1 \
  --optimizer_type=lamb \
  --scale_loss \
  --steps_before_eval_start=3948 \
  --steps_between_eval=658 \
  --steps_per_loop=658 \
  --stop_steps=8000 \
  --train_batch_size=760 \
  --verbosity=0 \
  --warmup_steps=420
  ```

### On TPU-V3-32

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
  sudo nvidia-docker run --rm \
  -v /home/kbuilder:/workspace \
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
  --scratch_gcs_url= \
  --root_data_dir=/data \
  --bigquery_project_name= \
  '--git_repos=<mlperf_repo>;<branch>;<commit hash>' \
  --data_downloads=gs://nnigania_perf_profiles/bert_mlperf_data \
  --python_path=staging/models/prod,mlcompass \
  --benchmark_methods=tf2_bert.bert_benchmark.BertClassifyBenchmarkReal.benchmark_4x4_tpu \
  --gcloud_key_file_url= gs://mlcompass-data/tokens/perfzero-tpu-credentials.json \
  --result_upload_methods=perfzero_export_lib.upload_summary '--tpu_parameters={\
      "name": "somename", \
      "project": "someproject", \
      "zone": "somezone", \
      "size": "v3-32", \
      "version": "2.3.0.dev20200620"}'
  ```

  The actual flags passed to the training script are:

  ```
  --bert_config_file=gs://nnigania_perf_profiles/bert_mlperf_data/bert_config.json \
  --beta_1=0.9 \
  --beta_2=0.93721 \
  --device_warmup \
  --distribution_strategy=tpu  \
  --do_eval \
  --dtype=bf16 \
  --eval_batch_size=768 \
  --init_checkpoint=gs://nnigania_perf_profiles/bert_mlperf_data/model.ckpt-28252 \
  '--input_files=gs://bert-data-europe/tfrecords3_500_parts/part-*' \
  --learning_rate=0.00047430 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --model_dir=/workspace/output/2020-06-25-18-15-51-461988/benchmark_4x4_tpu \
  --num_steps_per_epoch=5210 \
  --num_train_epochs=1 \
  --optimizer_type=lamb \
  --scale_loss \
  --steps_before_eval_start=3907 \
  --steps_between_eval=652 \
  --steps_per_loop=652 \
  --stop_steps=5210 \
  --tpu=perfzero-tpu-20200626-1012-9624 \
  --train_batch_size=768 \
  --verbosity=0 \
  --weight_decay_rate=0.0049573 \
  --warmup_steps=108 
  ```

## Submissions CLs

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
