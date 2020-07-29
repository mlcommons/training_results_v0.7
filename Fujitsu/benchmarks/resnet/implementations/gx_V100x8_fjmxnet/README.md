## Steps to launch training

### FUJITSU PRIMERGY GX2570M5 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2570M5
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2570M5:

```
LOGDIR=$LOGDIR DATADIR=$DATADIR BENCHMARK=$BENCHMARK NEXP=$NEXP ./run_and_time.sh
```

## Build MXNet
First, copy `config.mk` file from `mxnet/make` to `mxnet` and config your environment.
After that, launch `init_start.sh` file.

Please reffer to the following page:
https://mxnet.apache.org/get_started/build_from_source
