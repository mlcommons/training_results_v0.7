# 1. Problem

This problem uses recurrent neural network to do language translation.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)

# 2. Directions
## Steps to download and verify data
Download the data using the following command:

```
cd ..
bash download_dataset.sh
cd -
```

Verify data with:

```
cd ..
bash verify_dataset.sh
cd -
```
## Steps to launch training

### FUJITSU PRIMERGY GX2570M5 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2570M5
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2570M5:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-fujitsu:rnn_translator .
docker push <docker/registry>/mlperf-fujitsu:rnn_translator
```

2. Launch the training:

```
source config_PG.sh
CONT="<docker/registry>/mlperf-fujitsu:rnn_translator" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $PGNNODES -t $WALLTIME run.sub
```

