# 1. Problem

This task benchmarks reinforcement learning for the 19x19 version of the boardgame Go.
The model plays games against itself and uses these games to improve play.

## Requirements
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [TensorFlow (20.06-tf1-py3) NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)

# 2. Directions
### Steps to launch training for NVIDIA DGXA100 ( 32 nodes )
To setup the environment using nvidia-docker you can use the commands below.
To build tensorflow and minigo without nvidia-docker, please follow instructions at
https://github.com/tensorflow/minigo/tree/master/ml_perf/README.md


### Build docker and prepare dataset
```
    # Build a docker using Dockerfile in this directory
    nvidia-docker build -t mlperf-nvidia:minigo .

    # run docker
    nvidia-docker run -v <path/to/store/checkpoint>:/data --rm -it mlperf-nvidia:minigo
    cd minigo

    # Download dataset, needs gsutil.
    # Download & extract bootstrap checkpoint.
    gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
    tar xfz checkpoint.tar.gz -C ml_perf/

    # Download and freeze the target model.
    mkdir -p ml_perf/target/
    gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/

    # comment out L331 in dual_net.py before running freeze_graph.
    # L331 is: optimizer = hvd.DistributedOptimizer(optimizer)
    # Horovod is initialized via train_loop.py and isn't needed for this step.
    CUDA_VISIBLE_DEVICES=0 python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags  --model_path=ml_perf/target/target
    mv ml_perf/target/target.minigo ml_perf/target/target.minigo.tf

    # uncomment L331 in dual_net.py.
    # copy dataset to /data that is mapped to <path/to/store/checkpoint> outside of docker.
    # Needed because run_and_time.sh uses the following paths to load checkpoint
    # CHECKPOINT_DIR="/data/mlperf07"
    # TARGET_PATH="/data/target/target.minigo.tf"
    cp -a ml_perf/target /data/
    cp -a ml_perf/checkpoints/mlperf07 /data/

    # exit docker
```

### Run benchmark with SLURM for NVIDIA DGXA100 ( 32 nodes )
```
    # Data from DATADIR is mounted to /data/ in docker.
    # Launch configuration and system-specific hyperparameters for various system  are in config_DGXA100_multi_32nodes.sh. 
    CONT="mlperf-nvidia:minigo" DATADIR=<path/to/store/checkpoint> LOGDIR=<path/to/output/dir> DGXSYSTEM=DGXA100_multi_32nodes sbatch -N 32 -t $WALLTIME run.sub
```

All training data is generated during the selfplay phase of the RL loop.

The only data to be downloaded are the starting checkpoint and the target model. These are downloaded automatically
before the training starts.

# 3. Model
### Publication/Attribution

This benchmark is based on the [Minigo](https://github.com/tensorflow/minigo) project,
which is and inspired by the work done by Deepmind with
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961),
["Mastering the Game of Go without Human Knowledge"](https://www.nature.com/articles/nature24270), and
["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815).

Minigo is built on top of Brian Lee's [MuGo](https://github.com/brilee/MuGo), a pure Python
implementation of the first AlphaGo paper.

Note that Minigo is an independent effort from AlphaGo.

### Reinforcement Setup
This benchmark includes both the environment and training for 19x19 Go. There are three primary
parts to this benchmark.

 - Selfplay: the *latest trained* model plays games with itself as both black and white to produce
   board positions for training.
 - Training: waits for selfplay to play a specified number of games with the latest model, then
   trains the next model generation, updating the neural network waits. Selfplay constantly monitors
   the training output directory and loads the new weights when as they are produced by the trainer.
 - Target Evaluation: The training loop runs for a preset number of iterations, producing a new
   model generation each time. Once finished, target evaluation relplays the each trained model
   until it finds the first one that is able to beat a target model in at least 50% of the games.
   The time from training start to when this generation was produced is taken as the benchmark
   execution time.

### Structure
This task has a non-trivial network structure, including a search tree. A good overview of the
structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0.

### Weight and bias initialization and Loss Function
Network weights are initialized randomly. Initialization and loss are described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the network.


# 4. Quality

### Quality metric
Quality is measured by the number of games won out of 256 against a fixed target model.
The target model is downloaded before automatically before the training starts.

### Quality target
The target is to win at least 50% out of 256 games against the target model.

### Evaluation frequency
Evaluations are performed after completing the training and are not timed.
Checkpoints from every RL loop iteration are evaluated. 
