# mlperf-training SSD in mxnet

This is stripped out of [Gluon CV's](https://github.com/dmlc/gluon-cv) Model
Zoo, and then modified by Serge Panev to support some of the options we need to
match the SSD model in mlperf (Resnet-34 V1.5 backbone, with NHWC and fp16).

## Running code in Docker

### Launching the container

Use the following command to launch a prebuilt model container:

``
scripts/docker/launch.sh <DATASET_DIR> <RESULTS_DIR>
``

On the first execution of the script, docker will download the model image from NGC container registry.
The contents of this repository will be mounted to the `/workspace/ssd` directory inside the container.
Additionally, `<DATA_DIR>` and `<RESULT_DIR>` directories on the host will be mounted to `/datasets`, `/results` respectively.

### Building the image

This is an optional step and only applies if you want to build the image yourself, NGC container registry includes a prebuilt image.

To build the model image yourself, run the following script:

``
scripts/docker/build.sh
``

Note that the source and target image tags are defined in:

``
scripts/docker/config.sh
``

Make sure to change the target image name if you intended to upload your own image to NGC.


### Downloading the datasets and pretrained weights

Run the following script to download COCO-2017 dataset:

``
scripts/datasets/download_coco2017.sh
``

The compressed dataset will be downloaded to

``
/datasets/downloads
``

And the extracted files will be in:

``
/datasets/coco2017
``


### Download pretrained ResNet34 weights

From your **host**, execute the following script to obtain the ResNet34
pretrained weights:

``
scripts/datasets/get_resnet34_backbone.sh
``

The script will download ResNet34 weights from torchvision (`.pth` format) then
convert it to a `.pickle` file readable by mxnet. The script will automatically
download and run a PyTorch container for the conversion.


### Training the network

Use any of the scripts under

``
scripts/train/*
``

The script names should be self explanatory.

Please note that by default the training scripts expect the pretrained ResNet34
weights to be found at:

``
/datasets/backbones/resnet34-333f7ec4.pickle
``

### Dell EMC DSS8440 (single node)
Launch configuration and system-specific hyperparameters for the Dell EMC DSS8440
single node submission are in the `config_DSS8440.sh` script.

Steps required to launch single node training on Dell EMC:

```
docker build --pull -t mlperf-nvidia:image_classification .
CONT=mlperf-nvidia:image_classification DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PRETRAINED_DIR=<path/to/pretrained/dir> SYSTEM=DSS8440 ./run_with_docker.sh
```

### Dell EMC C4140 (single node and multi-node)

#### Steps to build Singularity container

```
bash ./build_container.sh
```

#### Steps to build OpenMPI compiler

The parallel SSD is implemented with Horovod library which uses MPI parallel programming model. So an MPI compiler is needed to run the benchmark. The MXNet 20.06 NGC container includes OpenMPI 3.1.6. The host needs to install an OpenMPI compiler with the version equal or greater than 3.1.6. By default OpenMPI 3.1.6 is used in the run script. You are free to change to other versions. 

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
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PRETRAINED_DIR=<path/to/pretrained/dir> SYSTEM=C4140 sbatch -N 1 -n 4 run.slurm
```
Run training on 2x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PRETRAINED_DIR=<path/to/pretrained/dir> SYSTEM=2xC4140 sbatch -N 2 -n 8 run.slurm
```
Run training on 4x C4140 server

```
DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PRETRAINED_DIR=<path/to/pretrained/dir> SYSTEM=4xC4140 sbatch -N 4 -n 16 run.slurm
```
