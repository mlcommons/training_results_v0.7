# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [MXNet 20.06-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) (multi-node)

# 2. Directions

## Steps to download and verify data
Download the data using the following command:

Please download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md#prepare-dataset .
