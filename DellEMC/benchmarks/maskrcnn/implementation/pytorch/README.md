# 1. Problem 
This benchmark uses Mask R-CNN for object detection.

## Requirements
* [PyTorch 19.05-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) (multi-node)

# 2. Directions

### Steps to download and verify data
The Mask R-CNN script operates on COCO, a large-scale object detection, segmentation, and captioning dataset.
To download and verify the dataset use following scripts:
   
    cd ~/reference/object_detection/
    ./download_dataset.sh
    ./verify_dataset.sh

This should return `PASSED`. 
To extract the dataset use:
   
    ./caffe2/extract_dataset.sh

Mask R-CNN uses pre-trained ResNet50 as a backbone. 
To download and verify the RN50 weights use:
 
    ./download_weights.sh

## Steps to launch training on a single node

For single-node training, we use docker to run our container.

### Dell EMC DSS8440
Launch configuration and system-specific hyperparameters for the Dell EMC DSS8440
single node submission are in the `config_DSS8440.sh` script.

Steps required to launch single node training on Dell EMC DSS8440:

```
docker build --pull -t mlperf-nvidia:object_detection .
CONT=mlperf-nvidia:object_detection DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> SYSTEM=DSS8440 ./run_with_docker.sh
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.


# 4. Model
### Publication/Attribution

We use a version of Mask R-CNN with a ResNet50 backbone. See the following papers for more background:

[1] [Mask R-CNN](https://arxiv.org/abs/1703.06870) by Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick, Mar 2017.

[2] [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.


### Structure & Loss
Refer to [Mask R-CNN](https://arxiv.org/abs/1703.06870) for the layer structure and loss function.


### Weight and bias initialization
The ResNet50 base must be loaded from the provided weights. They may be quantized.


### Optimizer
We use a SGD Momentum based optimizer with weight decay of 0.0001 and momentum of 0.9.


# 5. Quality
### Quality metric
As Mask R-CNN can provide both boxes and masks, we evaluate on both box and mask mAP.

### Quality target
Box mAP of 0.377, mask mAP of 0.339 

### Evaluation frequency
Once per epoch, 118k.

### Evaluation thoroughness
Evaluate over the entire validation set. Use the [NVIDIA COCO API](https://github.com/NVIDIA/cocoapi/) to compute mAP.
