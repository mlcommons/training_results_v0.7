# 1. Problem
This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [MXNet 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)

# 2. Directions
## Steps to download and verify data
Please download the dataset manually following the instructions from the ImageNet website. We use non-resized Imagenet dataset, packed into MXNet recordio database. It is not resized and not normalized. No preprocessing was performed on the raw ImageNet jpegs.

For further instructions, see https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/README.md# prepare-dataset .

## Setup
```
./setup.sh
```

## Steps to launch training
### NF5488
Launch configuration and system-specific hyperparameters for the NF5488 submission are in the config_NF5488.sh script.

Steps required to launch  training on NF5488:
```
cd ../implementation_closed
vim run_with_docker.sh
# Modify the host path of this line:
-v /home/zrg/mlperf/training_v0.7_v2/resnet/scripts:/workspace/image_classification \

CONT=mlperf-inspur:resnet DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir>  ./run_with_docker.sh
```


# 3. Dataset/Environment
## Publiction/Attribution.
We use Imagenet (http://image-net.org/):

O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet
large scale visual recognition challenge. arXiv:1409.0575, 2014.

## Training and test data separation
This is provided by the Imagenet dataset and original authors.

## Training data order
Each epoch goes over all the training data, shuffled every epoch.

## Test data order
We use validation dataset for evaluation. We don't provide an order of data traversal for evaluation.
# 4. Model

## Publication/Attribution
See the following papers for more background:

[1] Deep Residual Learning for Image Recognition by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] Identity Mappings in Deep Residual Networks by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.


# 5. Quality.

## Quality metric
Percent of correct classifications on the Image Net test dataset.

## Quality target
We run to 0.759 accuracy (75.9% correct classifications).

## Evaluation frequency
Every 4 epochs with offset 0 or 1 or 2 or 3.

## Evaluation thoroughness
Every test example is used each time.
