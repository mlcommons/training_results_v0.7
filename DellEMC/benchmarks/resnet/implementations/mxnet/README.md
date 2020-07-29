# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
### On Dell EMC DSS8440 server:
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [MXNet 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)


### On Dell EMC C4140 server(s)
* [MXNet 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)
* [Singularity container](https://github.com/sylabs/singularity)
* [OpenMPI compiler](https://www.open-mpi.org)
* [Slurm](https://www.schedmd.com/downloads.php)

# 2. Directions

## Steps to download and verify data
Download the data using the following command:

Please download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md#prepare-dataset .


## Steps to launch training 


### Dell EMC DSS8440 (single node)
Launch configuration and system-specific hyperparameters for the Dell EMC DSS8440
single node submission are in the `config_DSS8440.sh` script.

Steps required to launch single node training on Dell EMC DSS8440:

```
docker build --pull -t mlperf-nvidia:image_classification .
CONT=mlperf-nvidia:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=DSS8440 ./run_with_docker.sh
```

### Dell EMC C4140 (single node and multi-node)

#### Steps to build Singularity container

```
bash ./build_container.sh
```

#### Steps to build OpenMPI compiler

The parallel ResNet-50 is implemented with Horovod library which uses MPI parallel programming model. So an MPI compiler is needed to run the benchmark. The MXNet 20.06 NGC container includes OpenMPI 3.1.6. The host needs to install an OpenMPI compiler with the version equal or greater than 3.1.6. By default OpenMPI 3.1.6 is used in the run script. You are free to change to other versions. 

```
wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.6.tar.gz
tar xzfv openmpi-3.1.6.tar.gz && cd openmpi-3.1.6
mkdir build && cd build
../configure --prefix=</path/to/install> --with-cuda=</path/to/cuda11.0/toolkit>
make && make install
```

Then either create a module in your system, or add its bin and lib into PATH and LD_LIBRARY_PATH, respectively. 

### Steps to launch training

Launch configuration and system-specific hyperparameters for the Dell EMC 
single node and multi-node submission are in the `config_?C4140.sh` script.

Run training on single C4140 server:

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=C4140 sbatch -N 1 -n 4 run.slurm
```
Run training on 2x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=2xC4140 sbatch -N 2 -n 8 run.slurm
```
Run training on 4x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=4xC4140 sbatch -N 4 -n 16 run.slurm
```
