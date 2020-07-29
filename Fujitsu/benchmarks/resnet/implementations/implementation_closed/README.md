# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [MXNet 20.03-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)

# 2. Directions
## Steps to download and verify data
Download the data as follows:

Please download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md#prepare-dataset .

## Steps to launch training

### FUJITSU PRIMERGY GX2570M5 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2570M5
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2570M5:

```
cd ../gx_V100x8_closed
BENCHMARK=<benchmark_test> ./setup.sh
BENCHMARK=<benchmark_test> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PULL=0 ./run_and_time.sh
```





