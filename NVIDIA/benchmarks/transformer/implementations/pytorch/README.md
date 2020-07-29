# 1. Problem 

This problem uses Attention mechanisms to do language translation.

## Requirements

* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# 2. Directions

### Steps to download and verify data

Downloading and preprocessing the data is handled inside submission scripts. To do this manually run 
    bash run_preprocessing.sh && bash run_conversion.sh
    
The raw downloaded data is stored in /raw_data and preprocessed data is stored in /workspace/translation/examples/translation/wmt14_en_de. Your external DATADIR path can be mounted to this location to be used in the following steps. The vocabulary file provided by the MLPerf v0.7 transformer reference is stored inside of the container at /workspace/translation/reference_dictionary.ende.txt.

## Steps to launch training on a single node

For single-node training, we use docker to run our container.

### NVIDIA DGX A100 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
single node submission are in the `config_DGXA100.sh` script.

Steps required to launch single node training on NVIDIA DGX A100:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training:

```
source config_DGXA100.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX A100 (single node)](#nvidia-dgx-a100-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:translation .
source config_DGXA100.sh
CONT=mlperf-nvidia:translation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

### NVIDIA DGX-1 (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training:

```
source config_DGX1.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX-1 (single node)](#nvidia-dgx-1-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:translation .
source config_DGX1.sh
CONT=mlperf-nvidia:translation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

### NVIDIA DGX-2H (single node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
single node submission are in the `config_DGX2.sh` script.

Steps required to launch single node training on NVIDIA DGX-2H:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training:

```
source config_DGX2.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
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
docker build --pull -t mlperf-nvidia:translation .
source config_DGX2.sh
CONT=mlperf-nvidia:translation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

## Steps to launch training on multiple nodes

For multi-node training, we use Slurm for scheduling and Pyxis to run our container.

### NVIDIA DGX A100 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
multi node submission are in the following scripts:
* for the 2-node NVIDIA DGX A100 submission: `config_DGXA100_multi_2x8x6912.sh` 
* for the 10-node NVIDIA DGX A100 submission: `config_DGXA100_multi_10x8x9216.sh` 
* for the 20-node NVIDIA DGX A100 submission: `config_DGXA100_multi_20x8x4608.sh` 
* for the 60-node NVIDIA DGX A100 submission: `config_DGXA100_multi_60x8x1536.sh` 

Steps required to launch multi node training on NVIDIA DGX A100:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training:

2-node NVIDIA DGX A100 training:

```
source config_DGXA100_multi_2x8x6912.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

10-node NVIDIA DGX A100 training:

```
source config_DGXA100_multi_10x8x9216.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

20-node NVIDIA DGX A100 training:

```
source config_DGXA100_multi_20x8x4608.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

60-node NVIDIA DGX A100 training:

```
source config_DGXA100_multi_60x8x1536.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### NVIDIA DGX-2H (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the following scripts:
* for the 10-node NVIDIA DGX-2H submission: `config_DGX2_multi_10x16x4608.sh`
* for the 60-node NVIDIA DGX-2H submission: `config_DGX2_multi_60x16x768.sh`

Steps required to launch multi node training on NVIDIA DGX-2H:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training:

10-node NVIDIA DGX-2H training:

```
source config_DGX2_multi_10x16x4608.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

60-node NVIDIA DGX-2H training:

```
source config_DGX2_multi_60x16x768.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
