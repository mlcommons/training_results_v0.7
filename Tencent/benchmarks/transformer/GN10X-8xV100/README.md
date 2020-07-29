## Steps to launch training

### Tencent GN10X-8xV100 (single node)
Launch configuration and system-specific hyperparameters for the GN10X-8xV100
single node submission are in the `config_GN10X.sh` script.

Steps required to launch single node training:

```
cd ../implementations/pytorch
docker build --pull -t mlperf-nvidia:translation .
CONT=mlperf-nvidia:translation DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> DGXSYSTEM=GN10X ./run_with_docker.sh
```
