## Steps to launch training

### NVIDIA DGX-2H (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the `config_DGX2_multi_64x16x02x08.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2H

1. Build the docker container and push to a docker registry

```
cd ../implementations/mxnet
docker build --pull -t <docker/registry>/mlperf-nvidia:single_stage_detector-mxnet .
docker push <docker/registry>/mlperf-nvidia:single_stage_detector-mxnet
```

2. Launch the training

```
source config_DGX2_multi_64x16x02x08.sh
CONT="<docker/registry>/mlperf-nvidia:single_stage_detector-mxnet" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
