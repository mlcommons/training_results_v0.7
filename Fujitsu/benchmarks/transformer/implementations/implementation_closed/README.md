# 1. Problem 

This problem uses Attention mechanisms to do language translation.

## Requirements

* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

# 2. Directions

### Steps to download and verify data

Downloading and preprocessing the data is handled inside submission scripts. To do this manually run 
    bash run_preprocessing.sh && bash run_conversion.sh
    
The raw downloaded data is stored in /raw_data and preprocessed data is stored in /workspace/translation/examples/translation/wmt14_en_de. Your external DATADIR path can be mounted to this location to be used in the following steps. The vocabulary file provided by the MLPerf v0.7 transformer reference is stored inside of the container at /workspace/translation/reference_dictionary.ende.txt.

## Steps to launch training on a single node

For single-node training, we use docker to run our container.

### FUJITSU PRIMERGY GX2570M5 (single node)

Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2570M5
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2570M5:

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-fujitsu:translation .
docker push <docker/registry>/mlperf-fujitsu:translation
```

2. Launch the training:

```
source config_PG.sh
CONT="<docker/registry>/mlperf-fujitsu:translation" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $PGNNODES -t $WALLTIME run.sub
```

