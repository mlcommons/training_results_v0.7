## Steps to launch training on multiple nodes

For multi-node training, we use Slurm for scheduling and Pyxis to run our container.

### NVIDIA DGX A100 (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
multi node submission are in the following scripts:
* for the 4-node NVIDIA DGX A100 submission: `config_DGXA100_multi_4x8x4.sh` 

Steps required to launch multi node training on NVIDIA DGX A100:

1. Build the container:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:object_detection .
docker push <docker/registry>/mlperf-nvidia:object_detection
```

2. Launch the training:

4-node NVIDIA DGX A100 training:

```
source config_DGXA100_multi_4x8x4.sh
CONT="<docker/registry>/mlperf-nvidia:object_detection" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

