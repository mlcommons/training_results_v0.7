# 1. Problem
This problem uses recurrent neural network to do language translation.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 20.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch)

# 2. Directions
## Download the data using the following command:
```
./init_datasets.sh OUTPUT_DIR=${DST_DIR}
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
-v /home/zrg/mlperf/training_v0.7_v2/gnmt/scripts:/workspace/rnn_translator \

CONT=mlperf-inspur:rnn_translator DATADIR="${DST_DIR}" LOGDIR="${PWD}/logs" NFSYSTEM=NF5488 ./run_with_docker.sh
```

# 3. Dataset/Environment
## Publiction/Attribution.
We use WMT16 English-German for training.

## Training and test data separation
Training uses WMT16 English-German dataset, validation is on concatenation of newstest2015 and newstest2016, BLEU evaluation is done on newstest2014..

# 4. Model.
## Publication/Attribution
Implemented model is similar to the one from Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation paper.

Most important difference is in the attention mechanism. This repository implements gnmt_v2 attention: output from first LSTM layer of decoder goes into attention, then re-weighted context is concatenated with inputs to all subsequent LSTM layers in decoder at current timestep.

The same attention mechanism is also implemented in default GNMT-like models from tensorflow/nmt and NVIDIA/OpenSeq2Seq.

# 5. Quality.
## Quality metric
Uncased BLEU score on newstest2014 en-de dataset. BLEU scores reported by sacrebleu package (version 1.2.10). Sacrebleu is executed with the following flags: --score-only -lc --tokenize intl.

## Quality target
Uncased BLEU score of 24.00.

## Evaluation frequency
Evaluation of BLEU score is done after every epoch.

## Evaluation thoroughness
Evaluation uses all of newstest2014.en (3003 sentences).
