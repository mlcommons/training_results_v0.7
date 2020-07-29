# MLPerf Training v0.7 Intel Submission

# 1. Problem
This task benchmarks on policy reinforcement learning for the 9x9 version of the boardgame Go.
The model plays games against itself and uses these games to improve play.

# 2. Directions
### Steps to run MiniGo

```
    # Set WORKSPACE as /path/to/Intel/benchmarks/minigo/<system_desc_id>
    cd <WORKSPACE>

    # Install dependencies
    apt-get install -y python3 python3-pip rsync git wget pkg-config zip g++ zlib1g-dev unzip

    # Install Intel MPI 2018.1.163
    # Refer to the following commands to configure Intel MPI
    source /path/to/compilers_and_libraries/linux/bin/compilervars.sh intel64
    source /path/to/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh intel64

    # Install GCC 8.4.0
    # Refer to the following commands to configure GCC
    export PATH=path/to/gcc8.4.0/bin:$PATH
    export LD_LIBRARY_PATH=path/to/gcc8.4.0/lib64/:$LD_LIBRARY_PATH

    # download mlperf logging package
    git clone https://github.com/mlperf/logging.git mlperf-logging
    pip install -e mlperf-logging

    # Install anaconda and create an anaconda env (this step is optional but highly recommended).
    conda create -n minigo_xeon_opt python=3.6
    conda activate minigo_xeon_opt

    # Install Python dependencies
    pip3 install -r requirements.txt

    # Download & extract bootstrap checkpoint
    gsutil cp gs://minigo-pub/ml_perf/0.7/checkpoint.tar.gz .
    tar xfz checkpoint.tar.gz -C ml_perf/

    # Download and freeze the target model
    mkdir -p ml_perf/target/
    gsutil cp gs://minigo-pub/ml_perf/0.7/target.* ml_perf/target/
    # Note, freeze_graph.sh will install tensorflow-1.15 temporarily
    sh freeze_graph.sh

    # Install bazel 3.0.0
    BAZEL_VERSION=3.0.0
    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    chmod 755 bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    sh ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

    # Build and install specified Intel-Tensorflow
    cd <WORKSPACE>
    git clone https://github.com/Intel-tensorflow/tensorflow.git
    cd tensorflow
    git checkout 896a070312136fe944fd7a905e72f86dee9771f6
    # Apply compatibility patch for tensorflow-io
    git apply <WORKSPACE>/patches/compat.patch
    # Use all defaults
    ./configure
    bazel build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 --copt=-O3 --copt=-Wformat \
    --copt=-Wformat-security --copt=-fstack-protector --copt=-fPIC --copt=-fpic \
    --linkopt=-znoexecstack --linkopt=-zrelro --linkopt=-znow --linkopt=-fstack-protector \
    --config=mkl --define build_with_mkl_dnn_v1_only=true --copt=-DENABLE_INTEL_MKL_BFLOAT16 \
    --copt=-march=haswell //tensorflow/tools/pip_package:build_pip_package
    # Build and install python wheel
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ./tensorflow_pkg/
    pip install ./tensorflow_pkg/tensorflow-2.2.0-cp36-cp36m-linux_x86_64.whl

    cd <WORKSPACE>

    # Install Tensorflow-io
    pip3 install "tensorflow-io==0.13.0" --no-deps

    # Install horovod
    # Note, please ensure that Intel MPI and GCC have been configured correctly before compling horovod
    pip --no-cache-dir install horovod==0.19.1

    # Install bazel 0.24.1
    BAZEL_VERSION=0.24.1
    wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    chmod 755 bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
    sh ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user

    # Compile TensorFlow C++ libraries (enter to use default config)
    ./cc/configure_tensorflow.sh

    # Convert selfplay data format from NCHW to NHWC
    ./ml_perf/scripts/convert_data_format.sh

    # Configure nopassword on host machines
    # Launch the following command to run the workload
    HOSTLIST=<comma seperated hostlist> ml_perf/scripts/run_minigo.sh 19

    # Optionally, set host list use for train stage only
    HOSTLIST=<comma seperated hostlist> TRAINHOSTLIST=<comma seperated hostlist for training> ml_perf/scripts/run_minigo.sh 19
    # Note, $TRAINHOSTLIST could overlap with $HOSTLIST

    # Log post-processing
    # Running the benchmark creates two log files in <WORKSPACE>: train.log and eval.stdout
    # train.log captures MLPerf logging events, and eval.stdout contains std output from ml_perf/eval_models.py
    # Correct logging errors for compliance
    python postprocessing.py --in-file train.log --out-file train_postprocess.log
    # Update run_stop timestamp using the TTT from eval.stdout, and save as result.txt
    ./logging_postprocess.sh train_postprocess.log eval.stdout result.txt


    # Run MLPerf logging compliance checker
    python -m mlperf_logging.compliance_checker --config 0.7.0/closed_minigo.yaml --ruleset 0.7.0 result.txt
```

### Tunable hyperparameters

The following flags are allowed to be modified by entrants:
Flags that don't directly affect convergence:
 - _all flags related to file paths & device IDs_
 - `bool_features`
 - `input_layout`
 - `summary_steps`
 - `cache_size_mb`
 - `num_read_threads`
 - `num_write_threads`
 - `output_threads`
 - `selfplay_threads`
 - `parallel_search`
 - `parallel_inference`
 - `concurrent_games_per_thread`
 - `validate`
 - `holdout_pct`

Flags that directly affect convergence:
 - `train_batch_size`
 - `lr_rates`
 - `lr_boundaries`

Entrants are also free to replace the code responsible for writing, shuffling
and sampling training examples with equivalent functionality (e.g. replacing the
use of `sample_records` and a file system with a different storage solution).

### Selfplay threading model

The selfplay C++ binary (`//cc:concurrent_selfplay`) has multiple flags that control its
threading:

- `selfplay_threads` controls the number of threads that play selfplay games.
- `concurrent_games_per_thread` controls how many games are played on each thread. All games
  on a selfplay thread have their inferences batched together and dispatched at the same time.
- `parallel_search` controls the size of the thread pool shared between all selfplay threads
  that are used to parallelise tree search. Since the selfplay thread also performs tree
  search, the thread pool size is `parallel_search - 1` and a value of `1` disables the thread
  pool entirely.

To get a better understanding of the threading model, we recommend running a trace of the selfplay
code as described below.

### Profiling

The selfplay C++ binary can output traces of the host CPU using Google's
[Tracing Framework](https://google.github.io/tracing-framework/). Compile the
`//cc:concurrent_selfplay` binary with tracing support by passing `--copt=-DWTF_ENABLE` to
`bazel build`. Then run `//cc:concurrent_selfplay`, passing `--wtf_trace=$TRACE_PATH` to specify
the trace output path. The trace is appended to peridically, so the `//cc:concurrent_selfplay`
binary can be killed after 20 or so seconds and the trace file written so far will be valid.

Install the Tracing Framework [Chrome extension](https://google.github.io/tracing-framework/) to
view the trace.

Note that the amount of CPU time spent performing tree search changes over the lifetime of the
benchmark: initially, the models tend to read very deeply, which takes more CPU time.

Here is an example of building and running selfplay with tracing enabled on a single GPU:

```
bazel build -c opt --copt=-O3 --define=tf=1 --copt=-DWTF_ENABLE cc:concurrent_selfplay
CUDA_VISIBLE_DEVICES=0 ./bazel-bin/cc/concurrent_selfplay \
    --flagfile=ml_perf/flags/19/selfplay.flags \
    --wtf_trace=$HOME/mlperf07.wtf-trace \
    --model=$BASE_DIR/models/000001.minigo
```

### Steps to download and verify data
Unlike other benchmarks, there is no data to download. All training data comes from games played
during benchmarking.

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
Due to the difficulty of training a highly proficient Go model, our quality metric and termination
criteria is based on winning against a model of only intermediate amateur strength.

### Quality metric
The quality of a model is measured as the number of games won in a playoff (alternating colors)
of 100 games against a previously trained model.

### Quality target
The quality target is to win 50% of the games.

### Quality Progression
Informally, we have observed that quality should improve roughly linearly with time.  We observed
roughly 0.5% improvement in quality per hour of runtime. An example of approximately how we've seen
quality progress over time:

```
    Approx. Hours to Quality
     1h           TDB%
     2h           TDB%
     4h           TDB%
     8h           TDB%
```

Note that quality does not necessarily monotonically increase.

### Target evaluation frequency
Target evaluation only needs to be performed for models which pass model evaluation.
