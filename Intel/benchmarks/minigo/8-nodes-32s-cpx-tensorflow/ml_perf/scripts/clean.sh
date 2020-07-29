base_dir=$HOME/mlperf/results-$1

source ml_perf/scripts/common.sh

echo "Cleaning up temporary files."

if [[ -v HOSTLIST ]]; then
    mpirun -bootstrap ssh -ppn 1 -hosts $HOSTLIST sh ml_perf/scripts/rm_tmp_dir.sh \
        ${golden_chunk_local_dir}     \
        ${golden_chunk_tmp_dir} \
        ${selfplay_local_dir}   \
        ${local_signal_dir}
else
    sh ml_perf/scripts/rm_tmp_dir.sh \
        ${golden_chunk_local_dir}     \
        ${golden_chunk_tmp_dir} \
        ${selfplay_local_dir}   \
        ${local_signal_dir}
fi

echo "Done."
