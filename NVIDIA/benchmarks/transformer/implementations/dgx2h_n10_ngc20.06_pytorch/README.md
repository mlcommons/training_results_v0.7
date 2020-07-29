## Steps to launch training

### NVIDIA DGX-2H (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the `config_DGX2_multi_10x16x4608.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2H:

1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:translation .
docker push <docker/registry>/mlperf-nvidia:translation
```

2. Launch the training:

```
source config_DGX2_multi_10x16x4608.sh
CONT="<docker/registry>/mlperf-nvidia:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
