export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=1
export HOROVOD_CYCLE_TIME=0.2
export MXNET_OPTIMIZER_AGGREGATION_SIZE=54
