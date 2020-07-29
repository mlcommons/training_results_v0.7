# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [MXNet 19.05-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) (multi-node)

# 2. Directions

## Steps to download and verify data
Download the data using the following command:

Please download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md#prepare-dataset .

## Steps to launch training on a single node

For single-node training, we use docker to run our container.

### NVIDIA DGX-1 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

```
docker build --pull -t mlperf-nvidia:image_classification .
source config_DGX1.sh
CONT=mlperf-nvidia:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

### NVIDIA DGX-2 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
single node submission are in the `config_DGX2.sh` script.

Steps required to launch single node training on NVIDIA DGX-2:

```
docker build --pull -t mlperf-nvidia:image_classification .
source config_DGX2.sh
CONT=mlperf-nvidia:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```

## Steps to launch training on multiple nodes

For multi-node training, we use Slurm for scheduling and Pyxis to run our container.

### NVIDIA DGX-1 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-1
multi node submission are in the `config_DGX1_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-1:

1. Build the docker container and push to a docker registry
```
docker build --pull -t <docker/registry>/mlperf-nvidia:image_classification .
docker push <docker/registry>/mlperf-nvidia:image_classification
```

2. Launch the training
```
source config_DGX1_multi.sh
CONT="<docker/registry>/mlperf-nvidia:image_classification" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub
```

### NVIDIA DGX-2 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2
multi node submission are in the `config_DGX2_multi.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2:

1. Build the docker container and push to a docker registry
```
docker build --pull -t <docker/registry>/mlperf-nvidia:image_classification .
docker push <docker/registry>/mlperf-nvidia:image_classification
```

2. Launch the training
```
source config_DGX2_multi.sh
CONT="<docker/registry>/mlperf-nvidia:image_classification" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub
```
