# Set the benchmark output base directory.
BASE_DIR=$HOME/mlperf/results-$1

timestamp=$(date +%s)
timestamp=$((timestamp * 1000))
echo ":::MLLOG {\"namespace\": \"\", \"time_ms\": $timestamp, \"event_type\": \"INTERVAL_START\", \"key\": \"init_start\", \"value\": true, \"metadata\": {\"file\": \"run_minigo.sh\", \"lineno\": 6}}" >> train.log


# Bootstrap the training loop from the checkpoint.
# This step also builds the required C++ binaries.
# Bootstrapping is not considered part of the benchmark.
./ml_perf/scripts/init_from_checkpoint.sh \
    --board_size=$1 \
    --base_dir=$BASE_DIR \
    --checkpoint_dir=ml_perf/checkpoints/mlperf07

# launch all selfplay and standby
./ml_perf/scripts/run_mlperf_selfplay.sh $1

# Start the training loop. This is the point when benchmark timing starts.
# The training loop produces trains the first model generated from the
# bootstrap, then waits for selfplay to play games from the new model.
# When enough games have been played (see min_games_per_iteration in
# ml_perf/flags/19/train_loop.flags), a new model is trained using these
# games. This process repeats for a preset number of iterations (again,
# see ml_perf/flags/19/train_loop.flags).
# The train scripts terminates the selfplay jobs on exit by writing an
# "abort" file to the $BASE_DIR.
./ml_perf/scripts/train.sh \
     --board_size=$1 \
     --precision=$2 \
     --base_dir=$BASE_DIR

# evaluation
./ml_perf/scripts/run_mlperf_eval.sh $1  2>&1 | tee eval.stdout

# Clean up temporary directories
./ml_perf/scripts/clean.sh $1
