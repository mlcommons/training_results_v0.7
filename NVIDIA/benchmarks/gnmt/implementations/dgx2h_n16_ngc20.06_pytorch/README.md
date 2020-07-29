## Steps to launch training

### NVIDIA DGX-2H (multi node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX-2H
multi node submission are in the `config_DGX2_multi_16x16x32.sh` script.

Steps required to launch multi node training on NVIDIA DGX-2H:

1. Build the docker container and push to a docker registry
```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:rnn_translator .
docker push <docker/registry>/mlperf-nvidia:rnn_translator
```

2. Launch the training

```
source config_DGX2_multi_16x16x32.sh
CONT="<docker/registry>/mlperf-nvidia:rnn_translator" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME --ntasks-per-node $DGXNGPU run.sub
```
