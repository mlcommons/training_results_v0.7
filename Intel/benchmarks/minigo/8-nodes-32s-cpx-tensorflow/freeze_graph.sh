#Freeze target model with tensorflow-1.15
if [ -n $(pip freeze | grep -w "intel-tensorflow") ]; then
    pip uninstall intel-tensorflow -y
fi

{
    pip install tensorflow==1.15
} || {
    echo "Install tensorflow-1.15 failed. Please check your environment."
    exit
}

{
    sed -i 's/nhwc/nchw/' ml_perf/flags/19/architecture.flags
    python3 freeze_graph.py --flagfile=ml_perf/flags/19/architecture.flags  --model_path=ml_perf/target/target --work_dir=ml_perf/target
    sed -i 's/nchw/nhwc/' ml_perf/flags/19/architecture.flags
    pip uninstall tensorflow -y
} && {
    echo "Freeze target model success."
}
