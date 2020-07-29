# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* CUDA 10.1.234
* Open MPI 3.1.5

# 2. Directions
## Steps to download and verify data
Download the data using the following command:

Please download the dataset manually following the instructions from the [ImageNet website](http://image-net.org/download). We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

