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

# Train ResNet50-v1.5 with ImageNet 1K with TensorFlow

## 1. Description

This repository trains ResNet50-v1.5 for image classification on ImageNet 1K dataset using TensorFlow. Part of code is forked from:

https://github.com/IntelAI/models/tree/v1.6.1/models/image_recognition/tensorflow/resnet50v1_5/training

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

- TensorFlow
Method 1: build from source
Clone the intel-tensorflow publish repo, checkout the branch, and build from source.
```bash
git clone https://github.com/Intel-tensorflow/tensorflow.git
cd tensorflow
git checkout tf2_lars
```
Configure and build TF with oneDNN using the following bazel build command: 
```
bazel build --copt=-O3 --copt=-march=native --copt=-DENABLE_INTEL_MKL_BFLOAT16  --config=mkl --define build_with_mkl_dnn_v1_only=true -c opt //tensorflow/tools/pip_package:build_pip_package
```
- OpenMPI:
This submission was done using OpenMPI 2.1.1. Newer versions should also work. Please consult OpenMPI page for installation instructions.  

- Horovod
We tested with horovod 0.19.1 version. Horovod can be installed using the following commands:
```
pip install --no-cache-dir horovod==0.19.1 
```

Method 2: docker container approach (recommended because OpenMPI and Horovod has already been setup in the docker)
- Download the container
Intel has released a docker container featuring BF16 integration and ResNet50 lars optimizer in TensorFlow and is available via the following command.
```
docker pull intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-rn50-nightly
```

- MLPerf logging utility
The TF ResNet50 code requires mlperf logging repo.
Please make sure the following step is performed under the directory containing this README.md file.

```bash
git clone https://github.com/mlperf/logging.git
git checkout 71b2a076e9319c2dedb635dd8a34ef71e3a455e5
```

- Launch the container

After the above docker image is pulled and mlperf logging repo is downloaded, you can launch the docker container using a command similar to the following:
```
docker run -it --privileged  --name=mlperf-rn50  -v /dataset/:/dataset -v /MLPerfv0.7-code-base/:/workspace {docker_image_id}  /bin/bash
```
The /dataset refers to the imagenet dataset, the MLPerfv0.7-code-base refers to the MLPerf training v0.7 code. The docker image id can be found by "docker images" command.

With the above steps done, TF+MPI+Horovod environment should be setup correctly.
Note: the above "--privileged" is needed to avoid seeing the issue similar to "https://github.com/horovod/horovod/issues/653". If you do not have the privilege, suggest trying Method 1, i.e., running on bare-metal or without the container.

- Run the workload
The code is tested on an 8S Xeon platform having 28 cores in each socket. You would need to configure the OpenMP related parameters if you are not using identical platforms. Also, you need to modify the related path in `run\_and\_time.sh` before you run the code.

### 2.2 Dataset

The TF ResNet50-v1.5 model is trained with ImageNet 1K, a popular image classification dataset from ILSVRC challenge. The dataset can be downloaded from:

http://image-net.org/download-images

More dataset requirements can be found at:

https://github.com/mlperf/training/tree/master/image_classification#3-datasetenvironment

## 3. Training

### 3.1 Structure & Loss

In brief, this is a 50 layer v1 CNN. Refer to [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) for the layer structure and loss function.

### 3.2 Optimizer
The script supports both SGD optimizer and LARS optimizer. For best TTT, LARS optimizer should be used. 

### 3.3 Training commands

- Train on multiple CPU sockets with TensorFlow
```
cd mlperf_resnet
./run_and_time.sh
```
### 3.4 Post processing the training output log to pass MLPerf compliance 
When the above script finishes, resnet50v1.5\_\*.txt files would be generated.
Run the following command to post process the results (as an example)

```
./post-process.sh resnet50v1.5\_0.txt
```

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
