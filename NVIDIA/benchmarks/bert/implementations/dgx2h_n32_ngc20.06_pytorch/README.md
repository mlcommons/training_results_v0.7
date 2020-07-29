## Steps to launch training on multiple nodes

### NVIDIA DGX-2H (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the following scripts:
* for the 32-node NVIDIA DGX-2H submission: `config_DGX2_32x16x18x1.sh`

Steps required to launch multi node training on NVIDIA DGX-2H:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

2. Launch the training:

32-node NVIDIA DGX-2H training:

```
source config_DGX2_32x16x18x1.sh
CONT=mlperf-nvidia:language_model DATADIR=<path/to/datadir> DATADIR_PHASE2=<path/to/datadir_phase2> EVALDIR=<path/to/evaldir> CHECKPOINTDIR=<path/to/checkpointdir> CHECKPOINTDIR_PHASE1=<path/to/checkpointdir_phase1 sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
