## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to
run our container.

### NVIDIA DGX-1 (single node)

Launch configuration and system-specific hyperparameters for the
NVIDIA DGX-1 single node submission are in the `config_DGX1.sh` script.

Steps required to launch single node training on NVIDIA DGX-1:

1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch
```

2. Launch the training:

```
source config_DGX1.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-pytorch DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX-1 (single
node)](#nvidia-dgx-1-single-node) explain how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
cd ../implementations/pytorch
docker build --pull -t mlperf-nvidia:single_stage_detector-pytorch .
source config_DGX1.sh
CONT=mlperf-nvidia:single_stage_detector-pytorch DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
