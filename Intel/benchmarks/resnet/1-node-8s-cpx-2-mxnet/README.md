<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Train ResNet50-v1.5 with ImageNet 1K

## 1. Description

This repository trains ResNet50-v1.5 for image classification on ImageNet 1K. Part of code is forked from:

https://github.com/dmlc/gluon-cv/tree/master/scripts/classification/imagenet

See the following papers for more background about the model:

[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

## 2. Installation

Please follow the below steps to complete the setup of running environment.

```bash
mkdir -p $(pwd)/resnet
cd $(pwd)/resnet
```

### 2.1 Software

- Intel MKL
```bash
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16533/l_mkl_2020.1.217.tgz
cd l_mkl_2020.1.217
./install.sh
```

- MXNet

Clone the master branch of mxnet, and build from source (recommended).

```bash
git clone https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet
git checkout 68cb9555c4b4779aaae90e593b745270cbb59033
git submodule update --recursive --init
make USE_OPENCV=1 USE_MKLDNN=1 USE_BLAS=mkl USE_INTEL_PATH=/opt/intel/ -j
export PYTHONPATH=$(pwd)/resnet/incubator-mxnet/python
# if mxnet cannot find libiomp5.so when importing, please add the path of it to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin/:$LD_LIBRARY_PATH
# if mxnet cannot find opencv, please try to install it manually (for Ubuntu).
apt-get install libopencv-dev
```

- GluonCV

```bash
pip install gluoncv==0.7.0
```

- MLPerf logging utility

```bash
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
```

- Horovod

See [Multinode-training with Horovod](HOROVOD.md).

### 2.2 Dataset

The MXNet ResNet50-v1.5 model is trained with ImageNet 1K, a popular image classification dataset from ILSVRC challenge. The dataset can be downloaded from:

http://image-net.org/download-images

More dataset requirements can be found at:

https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment

For MXNet, the recommended data format is [RecordIO](http://mxnet.io/architecture/note_data_loading.html), which concatenates multiple examples into seekable binary files for better read efficiency. MXNet provides a tool called [im2rec.py](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) to convert individual images into `.rec` files.

To prepare a RecordIO file containing ImageNet data, we firstly need to create `.lst` files which consist of the labels and image paths. We assume that the original images are
already downloaded to `/data/imagenet/raw/train-jpeg` and `/data/imagenet/raw/val-jpeg`.

```bash
python /opt/mxnet/tools/im2rec.py --list --recursive train /data/imagenet/raw/train-jpeg
python /opt/mxnet/tools/im2rec.py --list --recursive val /data/imagenet/raw/val-jpeg
```

Next, we generate the `.rec` (RecordIO files with data) and `.idx` files. To obtain the best training accuracy, we do not preprocess the images when creating the RecordIO file.

```bash
python im2rec.py --pass-through --num-thread 40 train /data/imagenet/raw/train-jpeg
python im2rec.py --pass-through --num-thread 40 val /data/imagenet/raw/val-jpeg
```

## 3. Training

### 3.1 Structure & Loss

In brief, this is a 50 layer v1 CNN. Refer to [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) for the layer structure and loss function.

### 3.2 Optimizer

For training with relatively small batch size (< 2048), we use [NAG](https://mxnet.apache.org/api/python/docs/api/optimizer/index.html#mxnet.optimizer.NAG) based optimizer. The momentum and learning rate are scaled based on the batch size.

For training with large batch size (> 2048), we use LARS optimizer.

Parameters of both optimizers, such as `batch_size`, `num_epochs`, `warmup_epochs`, `lr`, `weight_decay`, `momentum`, etc, can be set by passing specific argument into the training script. For more optimizer specific params, please refer to the corresponding doc.

### 3.3 Training commands

- Train on multiple CPU sockets with MXNet-Horovod

See [Multinode-training with Horovod](HOROVOD.md#usage).

## 4. Quality

### 4.1 Quality metric

Percent of correct classifications on the ImageNet test dataset.

### 4.2 Quality target

0.749 accuracy (74.9% correct classifications) with TensorFlow.

**Update**: MLPerf 0.6 changed the target to 75.9%.

### 4.3 Evaluation frequency

Evaluate after every 4 epoch.

### 4.4 Evaluation thoroughness

Every test example is used each time.
