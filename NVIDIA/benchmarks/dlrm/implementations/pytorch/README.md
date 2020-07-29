## 1. Problem

This benchmark uses [DLRM](https://github.com/facebookresearch/dlrm/tree/mlperf) for recommendation.

## 2. Directions

### Command line

Code supports running hybrid parallel on 4, 8 and 16 GPUs. For example:

```shell
python -u -m torch.distributed.launch --use_env --nproc_per_node 8   scripts/dist_train.py --model_config dlrm/config/mlperf_40m.limit.json --dataset [DATADIR] --lr 24 --warmup_steps 2750 --decay_end_lr 0 --decay_steps 27772 --decay_start_step 49315 --batch_size 55296
```

### 2.1 NVIDIA DGX-2H (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
single node submission are in the `config_DGX2.sh` script.

Steps required to launch DLRM training with Pytorch on a single NVIDIA DGX-2H:


1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:recommendation .
docker push <docker/registry>/mlperf-nvidia:recommendation
```

2. Launch the train on Slurm

```
source config_DGX2.sh
CONT=<docker/registry>/mlperf-nvidia:recommendation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2H sbatch -N 1 run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX-2H (single node)](#nvidia-dgx-2h-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
source config_DGX2.sh
CONT=<docker/registry>mlperf-nvidia:recommendation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGX2H ./run_with_docker.sh
```

### 2.2 NVIDIA DGX A100 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
single node submission are in the `config_DGXA100.sh` script.

Steps required to launch DLRM training with Pytorch on a single NVIDIA DGX A100:


1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:recommendation .
docker push <docker/registry>/mlperf-nvidia:recommendation
```

2. Launch the train on Slurm

```
source config_DGXA100.sh
CONT=<docker/registry>/mlperf-nvidia:recommendation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGXA100 sbatch -N 1 run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX-A100 (single node)](#nvidia-dgx-a100-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
source config_DGXA100.sh
CONT=<docker/registry>mlperf-nvidia:recommendation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGXA100 ./run_with_docker.sh
```

## 3. Dataset/Environment

[Download Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) is used for the benchmark. The dataset is anonymized. It is preprocessed by the script reference provides. The up limit of row count of each embedding table is limited to 40 million per MLPerf rule.

We process the data further by:

```shell
# SRC_DIR should be the path to binary dataset processed by the reference script
python scripts/split_data.py --bin_data_root [SRC_DIR] --out_root [DST_DIR] --stage train
python scripts/split_data.py --bin_data_root [SRC_DIR] --out_root [DST_DIR] --stage test
```

## 4. Model

https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/

## 5. Quality

### Quality metric

[ROC-AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

### Quality target

AUC 0.8025.

### Evaluation frequency

20 evaluations per epoch.