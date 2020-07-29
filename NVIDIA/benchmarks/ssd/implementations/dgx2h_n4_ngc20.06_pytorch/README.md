## Steps to launch training on multiple nodes

### NVIDIA DGX-2H (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the `config_DGX2_multi_4x16x24.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2H

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-pytorch
```

2. Launch the training

```
source config_DGX2_multi_4x16x24.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-pytorch" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME $DGXNGPU run.sub
```
