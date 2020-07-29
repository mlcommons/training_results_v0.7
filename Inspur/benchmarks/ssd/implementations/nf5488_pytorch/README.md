# 1. Problem
Object detection.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)

# 2. Directions
## Steps to download data
```
DATADIR=../implementation_closed/coco2017 ./init_datasets.sh
```

## Setup
```
./setup.sh
```

## get resnet34 backbone
```
cd ../implementation_closed
nvidia-docker run -it --rm -v "$(PWD)":/workspace/single_stage_detector mlperf-inspur:ssd
cd coco2017
cp ../torch_to_numpy.py .
../get_resnet34_backbone.sh
```

## prepare bounding boxes files
```
python3 ../prepare-json.py --keep-keys annotations/instances_val2017.json annotations/bbox_only_instances_val2017.json
python3 ../prepare-json.py annotations/instances_train2017.json annotations/bbox_only_instances_train2017.json
exit
```

## Steps to launch training
### Inspur NF5488 
Launch configuration and system-specific hyperparameters for the NF5488 submission are in the config_NF5488.sh script.

Steps required to launch training on NF5488:
```
vim run_with_docker.sh
# Modify the host path of this line:
-v /home/zrg/mlperf/training_v0.7_v2/ssd/scripts:/workspace/single_stage_detector \

CONT=mlperf-inspur:ssd DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir>  ./run_with_docker.sh
```

# 3. Dataset/Environment
## Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

## Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
## Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ImageNet (from torchvision).

# 5. Quality.
## Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

## Quality target
mAP of 0.23

## Evaluation frequency
The model is evaluated at epochs 40, 50, 55,60, 65,70,75 and 80.

## Evaluation thoroughness
All the images in COCO 2017 val data set.
