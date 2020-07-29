# Steps to launch training
Submissions assume that you have:

1.  The user has an OBS storage bucket which is located in the same region as your ModelArts service.
2.  The user instance must have permissions to access ModelArts APIs.
3.  The project must have quota to create ModelArts training jobs for the submission.

## Dataset Preparation

1.  Download original data from image-net.org.
2.  Run `build_imagenet_data.py` to create TFRecords from ImageNet data.
3.  Run `split_tf_record.py` to split validation files to 1024.
4.  Upload generated tfrecord dataset to OBS.

## Run

1.  Upload the benchmark scripts to OBS.

2.  Once the dataset and scripts are ready, simply launch `main.py` as a training job on ModelArts with the number of nodes.

3.  Get result logs on OBS path specified in the training job.
