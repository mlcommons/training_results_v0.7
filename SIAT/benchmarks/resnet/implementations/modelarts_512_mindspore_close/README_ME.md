# Steps to launch training
Submissions assume that you have:

1.  The user has an OBS storage bucket which is located in the same region as your ModelArts service.
2.  The user instance must have permissions to access ModelArts APIs.
3.  The project must have quota to create ModelArts training jobs for the submission.

## Dataset Preparation

The following [script](https://github.com/mindspore-ai/mindspore/blob/master/example/cv_to_mindrecord/ImageNet_Similar_Perf/run_imagenet.sh)
was used to create MindRecord from ImageNet data using instructions in the
[README](https://github.com/mindspore-ai/mindspore/tree/master/example/cv_to_mindrecord/ImageNet_Similar_Perf).
MindRecord can be created directly from files downloaded from http://image-net.org/download.

## Run

1.  Upload the benchmark scripts to OBS.

2.  Once the dataset and scripts are ready, simply launch `train.py` as a training job on ModelArts with the number of nodes.

3.  Get result logs on OBS path specified in the training job.
