# 1. Problem

Single Stage Object Detection.

## Requirements
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)

# 2. Directions

### Steps to download data
```
cd ../gx_V100x8_ngc20.06_pytorch
DATADIR=<path/to/data/dir> ./init_datasets.sh
```

## Steps to launch training

### FUJITSU PRIMERGY GX2570M5 (single node)
Launch configuration and system-specific hyperparameters for the FUJITSU PRIMERGY GX2570M5
single node submission are in the `config_PG.sh` script.

Steps required to launch single node training on FUJITSU PRIMERGY GX2570M5:

```
cd ../gx_V100x8_ngc20.06_pytorch
BENCHMARK=<benchmark_test> ./setup.sh
BENCHMARK=<benchmark_test> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> PULL=0 ./run_and_time.sh
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_PG.sh` files for each configuration and in `run_and_time.sh`.

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ImageNet (from torchvision).

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.23

### Evaluation frequency
The model is evaluated at epochs 40, 50, 55, and then every 5th epoch.

### Evaluation thoroughness
All the images in COCO 2017 val data set.
