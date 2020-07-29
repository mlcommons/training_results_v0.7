# 1. Problem
This problem uses recurrent neural network to do language translation.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)

# 2. Directions
## Download the data using the following command:
## download data
```
./init_datasets.sh
```
## Unzip the data to the SRC_DIR

## Create and enter continer,Download processing script
```
docker run --gpus all -it --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v ${SRC_DIR}:/data nvcr.io/nvidia/pytorch:20.03-py3 bash
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install -e mlperf-logging
git clone https://github.com/facebookresearch/dlrm.git && cd dlrm && git checkout mlperf
```
## modify SRC_DIR and DST_DIR path in /workspace/dlrm/bench/run_and_time.sh(--raw-data-file=SRC_DIR,--processed-data-file=/DST_DIR/terabyte_processed.npz)

## Run the script to process the data (note: this script is  preprocessing + training. Once the training starts, the data preprocessing has been completed and the data set can be saved for later use）
```
cd /workspace/dlrm
./bench/run_and_time.sh
```
## Generate the DLRM final dataset
```
python split_data.py --bin_data_root SRC_DIR --out_root DST_DIR --stage train 
python split_data.py --bin_data_root SRC_DIR --out_root DST_DIR --stage test

```

## Steps to launch training
### Inspur NF5488 
Launch configuration and system-specific hyperparameters for the Inspur NF5488 submission are in the config_NF5488.sh script.

Steps required to launch training on Inspur NF5488:
```
./setup.sh

cd ../implementation_closed
vim run_with_docker.sh

Modify the host path of this line:
-v /home/zrg/mlperf/training_v0.7_v2/dlrm/scripts:/workspace/dlrm \

CONT=mlperf-inspur:recommendation DATADIR="${DST_DIR}" LOGDIR="${PWD}/logs" NFSYSTEM=NF5488 ./run_with_docker.sh

```

# 3. Dataset/Environment
## Publiction/Attribution.
We use criteo-1tb-click-logs-dataset for training.

# 4. Model.
## Publication/Attribution
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). Neural Collaborative Filtering. In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The author's original code is available at hexiangnan/neural_collaborative_filtering.
# 5. Quality.
## Quality target
Hit target accuracy AUC 0.8025

## Evaluation frequency
Evaluation of BLEU score is done after 3792 steps
