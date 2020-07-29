# Set the benchmark output base directory.
BASE_DIR=$HOME/mlperf/results-$1

# Start the selfplay binaries running on the specified GPU devices.
# This launches one selfplay binary per GPU.
# This script can be run on multiple machines if desired, so long as all
# machines have access to the $BASE_DIR.
# In this particular example, the machine running the benchmark has 8 GPUs,
# of whice devices 1-7 are used for selfplay and device 0 is used for
# training.
# The selfplay jobs will start, and wait for the training loop (started in
# the next step) to produce the first model. The selfplay jobs run forever,
# reloading new generations of models as the training job trains them.
if [[ -v HOSTLIST ]]
then
mpirun -bootstrap ssh -ppn 1 -hosts $HOSTLIST ./ml_perf/scripts/start_selfplay.sh \
     --board_size=$1 \
     --base_dir=$BASE_DIR 2>&1 >/dev/null &
else
./ml_perf/scripts/start_selfplay.sh \
     --board_size=$1 \
     --base_dir=$BASE_DIR
fi

./ml_perf/scripts/start_sync_selfplay.sh --base_dir=$BASE_DIR
