## Steps to launch training on multiple nodes

### NVIDIA DGX-2H (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the following scripts:
* for the 16-node NVIDIA DGX-2H submission: `config_DGX2_multi.sh`

Steps required to launch multi node training on NVIDIA DGX-2H:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:object_detection .
docker push <docker/registry>/mlperf-nvidia:object_detection
```

2. Launch the training:

16-node NVIDIA DGX-2H training:

```
source config_DGX2_multi.sh
CONT="<docker/registry>/mlperf-nvidia:object_detection" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

