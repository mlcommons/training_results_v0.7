#!/bin/bash

set -ex

resnet50-v1.5_single_ins_8280_fp32() {

    set -ex
    pushd .

    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    export OMP_NUM_THREADS=28

    numactl --physcpubind=0-27 --membind=0 python train_imagenet_symbolic.py \
        --rec-train /lustre/dataset/mxnet/imagenet/train.rec \
        --rec-train-idx /lustre/dataset/mxnet/imagenet/train.idx \
        --rec-val /lustre/dataset/mxnet/imagenet/val.rec \
        --rec-val-idx /lustre/dataset/mxnet/imagenet/val.idx \
        --model resnet50_v1b --label-smoothing 0.1 --use-symbolic --optimizer "sgdwfastlars" \
        --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 0 \
        --warmup-epochs 5 --use-rec --log-interval 1

    popd
}

resnet50-v1.5_single_ins_cpx_bf16() {

    set -ex
    pushd .

    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    export OMP_NUM_THREADS=24

    numactl --physcpubind=0-23 --membind=0 python train_imagenet_symbolic.py \
        --rec-train /lustre/dataset/mxnet/imagenet/train.rec \
        --rec-train-idx /lustre/dataset/mxnet/imagenet/train.idx \
        --rec-val /lustre/dataset/mxnet/imagenet/val.rec \
        --rec-val-idx /lustre/dataset/mxnet/imagenet/val.idx \
        --model resnet50_v1b --amp --log-interval 2 \
        --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 0 \
        --warmup-epochs 5 --log-interval 1 --use-symbolic \
        --use-rec --label-smoothing 0.1

    popd
}

resnet50-v1.5_single_ins_6148_fp32() {

    set -ex
    pushd .

    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    export OMP_NUM_THREADS=20

    numactl --physcpubind=20-39 --membind=1 python train_imagenet_symbolic.py \
        --rec-train /lustre/dataset/mxnet/imagenet/train.rec \
        --rec-train-idx /lustre/dataset/mxnet/imagenet/train.idx \
        --rec-val /lustre/dataset/mxnet/imagenet/val.rec \
        --rec-val-idx /lustre/dataset/mxnet/imagenet/val.idx \
        --model resnet50_v1b --use-symbolic \
        --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 0 \
        --warmup-epochs 5 --log-interval 1 \
        --use-rec --label-smoothing 0.1

    popd
}

resnet50-v1.5_subset_single_ins_6148_fp32() {

    set -ex
    pushd .

    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    export OMP_NUM_THREADS=20

    numactl --physcpubind=20-39 --membind=1 python train_imagenet_symbolic.py \
        --rec-train /lustre/dataset/mxnet/tiny-imagenet/sub_train.rec \
        --rec-train-idx /lustre/dataset/mxnet/tiny-imagenet/sub_train.idx \
        --rec-val /lustre/dataset/mxnet/tiny-imagenet/sub_val.rec \
        --rec-val-idx /lustre/dataset/mxnet/tiny-imagenet/sub_val.idx \
        --random-seed 1234 --model resnet50_v1b \
        --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 0 \
        --warmup-epochs 5 --use-symbolic \
        --use-rec --label-smoothing 0.1

    popd
}

resnet50-v1.5_single_ins_6148_bf16() {

    set -ex
    pushd .

    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    export OMP_NUM_THREADS=20

    numactl --physcpubind=20-39 --membind=1 python train_imagenet_symbolic.py \
        --rec-train /lustre/dataset/mxnet/imagenet/train.rec \
        --rec-train-idx /lustre/dataset/mxnet/imagenet/train.idx \
        --rec-val /lustre/dataset/mxnet/imagenet/val.rec \
        --rec-val-idx /lustre/dataset/mxnet/imagenet/val.idx \
        --model resnet50_v1b --amp --use-symbolic \
        --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 0 \
        --warmup-epochs 5 --log-interval 1 \
        --use-rec --label-smoothing 0.1

    popd
}

resnet50-v1.5_single_ins_v100_fp32() {

    set -ex
    pushd .

    python train_imagenet_symbolic.py \
        --rec-train /lustre/dataset/mxnet/imagenet/train.rec \
        --rec-train-idx /lustre/dataset/mxnet/imagenet/train.idx \
        --rec-val /lustre/dataset/mxnet/imagenet/val.rec \
        --rec-val-idx /lustre/dataset/mxnet/imagenet/val.idx \
        --model resnet50_v1b \
        --lr 0.05 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 1 \
        --warmup-epochs 5 \
        --use-rec --label-smoothing 0.1

    popd
}

resnet50-v1.5_multi_node_fp32() {

    set -ex
    pushd .

    MXNET_DIR=$1
    DATA_DIR=$2
    export LD_LIBRARY_PATH=${MXNET_DIR}/lib:${LD_LIBRARY_PATH}
    export PYTHONPATH=${MXNET_DIR}/python

    phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')
    horovod_background_affinity_rank_0=$[${phy_cpus} * 2]
    horovod_background_affinity_rank_1=$[${phy_cpus} * 3]

    export MLSL_NUM_SERVERS=1
    export MLSL_SERVER_AFFINITY=0,${phy_cpus}
    if [ ${MPI_LOCALRANKID} -eq 0 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_0}
    fi
    if [ ${MPI_LOCALRANKID} -eq 1 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_1}
    fi

    node_count=$3
    instance_count=$4
    seed=$5

    python train_imagenet_symbolic.py \
            --model "resnet50_v1b" \
            --batch-size 128 \
            --num-epochs 42 \
            --warmup-epochs 5 \
            --lr 10 \
            --lr-step-epochs "pow2" \
            --lars-eta 0.001 \
            --lars-eps 0 \
            --log-interval 20 \
            --eval-frequency 4 \
            --eval-offset 3 \
            --use-rec \
            --rec-train $DATA_DIR/train.rec \
            --rec-train-idx $DATA_DIR/train.idx \
            --rec-val $DATA_DIR/val.rec \
            --rec-val-idx $DATA_DIR/val.idx \
            --random-seed $seed \
            --wd 0.0002 \
            --optimizer "sgdwfastlars" \
            --label-smoothing 0.1 \
            --momentum 0.9 \
            --num-examples 1281167 \
            --accuracy-target "0.759" \
            --use-symbolic \
            --horovod

    popd
}

resnet50-v1.5_multi_node_bf16() {

    set -ex
    pushd .

    MXNET_DIR=$1
    DATA_DIR=$2
    export LD_LIBRARY_PATH=${MXNET_DIR}/lib:${LD_LIBRARY_PATH}
    export PYTHONPATH=${MXNET_DIR}/python

    phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')
    horovod_background_affinity_rank_0=$[${phy_cpus} * 2]
    horovod_background_affinity_rank_1=$[${phy_cpus} * 3]

    export MLSL_NUM_SERVERS=1
    export MLSL_SERVER_AFFINITY=0,${phy_cpus}
    if [ ${MPI_LOCALRANKID} -eq 0 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_0}
    fi
    if [ ${MPI_LOCALRANKID} -eq 1 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_1}
    fi

    node_count=$3
    instance_count=$4
    seed=$5

    python train_imagenet_symbolic.py \
            --model resnet50_v1b \
            --batch-size 51 \
            --num-epochs 41 \
            --warmup-epochs 3 \
            --lr 3 \
            --lr-step-epochs "pow2" \
            --lars-eta 0.001 \
            --lars-eps 0 \
            --log-interval 20 \
            --eval-frequency 4 \
            --eval-offset 3 \
            --use-rec \
            --rec-train $DATA_DIR/train.rec \
            --rec-train-idx $DATA_DIR/train.idx \
            --rec-val $DATA_DIR/val.rec \
            --rec-val-idx $DATA_DIR/val.idx \
            --random-seed $seed \
            --wd 0.000025 \
            --optimizer "sgd" \
            --label-smoothing 0.1 \
            --momentum 0.9 \
            --num-examples 1281167 \
            --accuracy-target "0.759" \
            --use-symbolic \
            --horovod \
            --amp

    popd
}

resnet50-v1.5_multi_ins_cpx_fp32() {

    set -ex
    pushd .

    MXNET_DIR=$1
    DATA_DIR=$2

    export LD_LIBRARY_PATH=${MXNET_DIR}/lib:/opt/intel/compilers_and_libraries/linux/lib/intel64:${LD_LIBRARY_PATH}
    export PYTHONPATH=${MXNET_DIR}/python

    phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')

    horovod_background_affinity_rank_0=$[$phy_cpus * 4]
    horovod_background_affinity_rank_1=$[$phy_cpus * 5]
    horovod_background_affinity_rank_2=$[$phy_cpus * 6]
    horovod_background_affinity_rank_3=$[$phy_cpus * 7]

    export MLSL_NUM_SERVERS=1
    export MLSL_SERVER_AFFINITY=0,${phy_cpus},$[${phy_cpus}*2],$[${phy_cpus}*3]
    if [ ${MPI_LOCALRANKID} -eq 0 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_0}
    fi
    if [ ${MPI_LOCALRANKID} -eq 1 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_1}
    fi
    if [ ${MPI_LOCALRANKID} -eq 2 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_2}
    fi
    if [ ${MPI_LOCALRANKID} -eq 3 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_3}
    fi

    node_count=$3
    instance_count=$4
    seed=$5

    python train_imagenet_symbolic.py \
            --model "resnet50_v1b" \
            --batch-size 128 \
            --num-epochs 42 \
            --warmup-epochs 5 \
            --lr 10 \
            --lr-step-epochs "pow2" \
            --lars-eta 0.001 \
            --lars-eps 0 \
            --log-interval 20 \
            --eval-frequency 4 \
            --eval-offset 3 \
            --use-rec \
            --rec-train $DATA_DIR/train/train.rec \
            --rec-train-idx $DATA_DIR/train/train.idx \
            --rec-val $DATA_DIR/val/val.rec \
            --rec-val-idx $DATA_DIR/val/val.idx \
            --random-seed $seed \
            --wd 0.0002 \
            --optimizer "sgdwfastlars" \
            --label-smoothing 0.1 \
            --momentum 0.9 \
            --num-examples 1281167 \
            --accuracy-target "0.759" \
            --use-symbolic \
            --horovod

    popd
}

resnet50-v1.5_multi_ins_cpx_bf16() {

    set -ex
    pushd .

    MXNET_DIR=$1
    DATA_DIR=$2

    export LD_LIBRARY_PATH=${MXNET_DIR}/lib:/opt/intel/compilers_and_libraries/linux/lib/intel64:${LD_LIBRARY_PATH}
    export PYTHONPATH=${MXNET_DIR}/python

    phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')

    horovod_background_affinity_rank_0=$[$phy_cpus * 4]
    horovod_background_affinity_rank_1=$[$phy_cpus * 5]
    horovod_background_affinity_rank_2=$[$phy_cpus * 6]
    horovod_background_affinity_rank_3=$[$phy_cpus * 7]

    export MLSL_NUM_SERVERS=1
    export MLSL_SERVER_AFFINITY=0,${phy_cpus},$[${phy_cpus}*2],$[${phy_cpus}*3]
    if [ ${MPI_LOCALRANKID} -eq 0 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_0}
    fi
    if [ ${MPI_LOCALRANKID} -eq 1 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_1}
    fi
    if [ ${MPI_LOCALRANKID} -eq 2 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_2}
    fi
    if [ ${MPI_LOCALRANKID} -eq 3 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_3}
    fi

    node_count=$3
    instance_count=$4
    seed=$5

    python train_imagenet_symbolic.py \
            --model "resnet50_v1b" \
            --batch-size 128 \
            --num-epochs 42 \
            --warmup-epochs 5 \
            --lr 10 \
            --lr-step-epochs "pow2" \
            --lars-eta 0.001 \
            --lars-eps 0 \
            --log-interval 20 \
            --eval-frequency 4 \
            --eval-offset 3 \
            --use-rec \
            --rec-train $DATA_DIR/train/train.rec \
            --rec-train-idx $DATA_DIR/train/train.idx \
            --rec-val $DATA_DIR/val/val.rec \
            --rec-val-idx $DATA_DIR/val/val.idx \
            --random-seed $seed \
            --wd 0.0002 \
            --optimizer "sgdwfastlars" \
            --label-smoothing 0.1 \
            --momentum 0.9 \
            --num-examples 1281167 \
            --accuracy-target "0.759" \
            --use-symbolic \
            --horovod \
            --amp

    popd
}

resnet50-v1.5_multi_ins_hyve_bf16() {

    set -ex
    pushd .

    MXNET_DIR=$1
    DATA_DIR=$2

    export LD_LIBRARY_PATH=${MXNET_DIR}/lib:${LD_LIBRARY_PATH}
    export PYTHONPATH=${MXNET_DIR}/python

    phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')
    interval=7
    
    horovod_background_affinity_rank_0=$[($phy_cpus * 8) + ($interval * 0)]
    horovod_background_affinity_rank_1=$[($phy_cpus * 8) + ($interval * 1)]
    horovod_background_affinity_rank_2=$[($phy_cpus * 8) + ($interval * 2)]
    horovod_background_affinity_rank_3=$[($phy_cpus * 8) + ($interval * 3)]
    horovod_background_affinity_rank_4=$[($phy_cpus * 8) + ($interval * 4)]
    horovod_background_affinity_rank_5=$[($phy_cpus * 8) + ($interval * 5)]
    horovod_background_affinity_rank_6=$[($phy_cpus * 8) + ($interval * 6)]
    horovod_background_affinity_rank_7=$[($phy_cpus * 8) + ($interval * 7)]
    horovod_background_affinity_rank_8=$[($phy_cpus * 8) + ($interval * 8)]
    horovod_background_affinity_rank_9=$[($phy_cpus * 8) + ($interval * 9)]
    horovod_background_affinity_rank_10=$[($phy_cpus * 8) + ($interval * 10)]
    horovod_background_affinity_rank_11=$[($phy_cpus * 8) + ($interval * 11)]
    horovod_background_affinity_rank_12=$[($phy_cpus * 8) + ($interval * 12)]
    horovod_background_affinity_rank_13=$[($phy_cpus * 8) + ($interval * 13)]
    horovod_background_affinity_rank_14=$[($phy_cpus * 8) + ($interval * 14)]
    horovod_background_affinity_rank_15=$[($phy_cpus * 8) + ($interval * 15)]
    horovod_background_affinity_rank_16=$[($phy_cpus * 8) + ($interval * 16)]
    horovod_background_affinity_rank_17=$[($phy_cpus * 8) + ($interval * 17)]
    horovod_background_affinity_rank_18=$[($phy_cpus * 8) + ($interval * 18)]
    horovod_background_affinity_rank_19=$[($phy_cpus * 8) + ($interval * 19)]
    horovod_background_affinity_rank_20=$[($phy_cpus * 8) + ($interval * 20)]
    horovod_background_affinity_rank_21=$[($phy_cpus * 8) + ($interval * 21)]
    horovod_background_affinity_rank_22=$[($phy_cpus * 8) + ($interval * 22)]
    horovod_background_affinity_rank_23=$[($phy_cpus * 8) + ($interval * 23)]
    horovod_background_affinity_rank_24=$[($phy_cpus * 8) + ($interval * 24)]
    horovod_background_affinity_rank_25=$[($phy_cpus * 8) + ($interval * 25)]
    horovod_background_affinity_rank_26=$[($phy_cpus * 8) + ($interval * 26)]
    horovod_background_affinity_rank_27=$[($phy_cpus * 8) + ($interval * 27)]
    horovod_background_affinity_rank_28=$[($phy_cpus * 8) + ($interval * 28)]
    horovod_background_affinity_rank_29=$[($phy_cpus * 8) + ($interval * 29)]
    horovod_background_affinity_rank_30=$[($phy_cpus * 8) + ($interval * 30)]
    horovod_background_affinity_rank_31=$[($phy_cpus * 8) + ($interval * 31)]

    export MLSL_NUM_SERVERS=1
    export MLSL_SERVER_AFFINITY=0,${interval},$[${interval}*2],$[${interval}*3],$[${interval}*4],$[${interval}*5],$[${interval}*6],$[${interval}*7],$[${interval}*8],$[${interval}*9],$[${interval}*10],$[${interval}*11],$[${interval}*12],$[${interval}*13],$[${interval}*14],$[${interval}*15],$[${interval}*16],$[${interval}*17],$[${interval}*18],$[${interval}*19],$[${interval}*20],$[${interval}*21],$[${interval}*22],$[${interval}*23],$[${interval}*24],$[${interval}*25],$[${interval}*26],$[${interval}*27],$[${interval}*28],$[${interval}*29],$[${interval}*30],$[${interval}*31]
    if [ ${MPI_LOCALRANKID} -eq 0 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_0}
    fi
    if [ ${MPI_LOCALRANKID} -eq 1 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_1}
    fi
    if [ ${MPI_LOCALRANKID} -eq 2 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_2}
    fi
    if [ ${MPI_LOCALRANKID} -eq 3 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_3}
    fi
    if [ ${MPI_LOCALRANKID} -eq 4 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_4}
    fi
    if [ ${MPI_LOCALRANKID} -eq 5 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_5}
    fi
    if [ ${MPI_LOCALRANKID} -eq 6 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_6}
    fi
    if [ ${MPI_LOCALRANKID} -eq 7 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_7}
    fi
    if [ ${MPI_LOCALRANKID} -eq 8 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_8}
    fi
    if [ ${MPI_LOCALRANKID} -eq 9 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_9}
    fi
    if [ ${MPI_LOCALRANKID} -eq 10 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_10}
    fi
    if [ ${MPI_LOCALRANKID} -eq 11 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_11}
    fi
    if [ ${MPI_LOCALRANKID} -eq 12 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_12}
    fi
    if [ ${MPI_LOCALRANKID} -eq 13 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_13}
    fi
    if [ ${MPI_LOCALRANKID} -eq 14 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_14}
    fi
    if [ ${MPI_LOCALRANKID} -eq 15 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_15}
    fi
    if [ ${MPI_LOCALRANKID} -eq 16 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_16}
    fi
    if [ ${MPI_LOCALRANKID} -eq 17 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_17}
    fi
    if [ ${MPI_LOCALRANKID} -eq 18 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_18}
    fi
    if [ ${MPI_LOCALRANKID} -eq 19 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_19}
    fi
    if [ ${MPI_LOCALRANKID} -eq 20 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_20}
    fi
    if [ ${MPI_LOCALRANKID} -eq 21 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_21}
    fi
    if [ ${MPI_LOCALRANKID} -eq 22 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_22}
    fi
    if [ ${MPI_LOCALRANKID} -eq 23 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_23}
    fi
    if [ ${MPI_LOCALRANKID} -eq 24 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_24}
    fi
    if [ ${MPI_LOCALRANKID} -eq 25 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_25}
    fi
    if [ ${MPI_LOCALRANKID} -eq 26 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_26}
    fi
    if [ ${MPI_LOCALRANKID} -eq 27 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_27}
    fi
    if [ ${MPI_LOCALRANKID} -eq 28 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_28}
    fi
    if [ ${MPI_LOCALRANKID} -eq 29 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_29}
    fi
    if [ ${MPI_LOCALRANKID} -eq 30 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_30}
    fi
    if [ ${MPI_LOCALRANKID} -eq 31 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_31}
    fi

    node_count=$3
    instance_count=$4
    seed=$5

    python train_imagenet_symbolic.py \
            --model "resnet50_v1b" \
            --batch-size 128 \
            --num-epochs 42 \
            --warmup-epochs 5 \
            --lr 10 \
            --lr-step-epochs "pow2" \
            --lars-eta 0.001 \
            --lars-eps 0 \
            --log-interval 50 \
            --eval-frequency 4 \
            --eval-offset 3 \
            --use-rec \
            --rec-train $DATA_DIR/train/train.rec \
            --rec-train-idx $DATA_DIR/train/train.idx \
            --rec-val $DATA_DIR/val/val.rec \
            --rec-val-idx $DATA_DIR/val/val.idx \
            --random-seed $seed \
            --wd 0.0002 \
            --optimizer "sgdwfastlars" \
            --label-smoothing 0.1 \
            --momentum 0.9 \
            --num-examples 1281167 \
            --accuracy-target "0.759" \
            --use-symbolic \
            --horovod \
            --amp

    popd
}

resnet50-v1.5_multi_ins_rome_fp32() {

    set -ex
    pushd .

    MXNET_DIR=$1
    DATA_DIR=$2

    export LD_LIBRARY_PATH=${MXNET_DIR}/lib:/opt/intel/compilers_and_libraries/linux/lib/intel64:${LD_LIBRARY_PATH}
    export PYTHONPATH=${MXNET_DIR}/python

    phy_cpus=$(lscpu | grep 'Core(s) per socket' | awk '{print$4}')

    # pin horovod background thread to hyper-threading cores (hard-coded)
    horovod_background_affinity_rank_0=128
    horovod_background_affinity_rank_1=144
    horovod_background_affinity_rank_2=160
    horovod_background_affinity_rank_3=176
    horovod_background_affinity_rank_4=192
    horovod_background_affinity_rank_5=208
    horovod_background_affinity_rank_6=224
    horovod_background_affinity_rank_7=240

    export MLSL_NUM_SERVERS=1
    # pin mlsl server to the first core of each numa node
    export MLSL_SERVER_AFFINITY=0,16,32,48,64,80,96,112
    if [ ${MPI_LOCALRANKID} -eq 0 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_0}
    fi
    if [ ${MPI_LOCALRANKID} -eq 1 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_1}
    fi
    if [ ${MPI_LOCALRANKID} -eq 2 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_2}
    fi
    if [ ${MPI_LOCALRANKID} -eq 3 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_3}
    fi
    if [ ${MPI_LOCALRANKID} -eq 4 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_4}
    fi
    if [ ${MPI_LOCALRANKID} -eq 5 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_5}
    fi
    if [ ${MPI_LOCALRANKID} -eq 6 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_6}
    fi
    if [ ${MPI_LOCALRANKID} -eq 7 ]; then
        export HOROVOD_MLSL_BGT_AFFINITY=${horovod_background_affinity_rank_7}
    fi

    node_count=$3
    instance_count=$4
    seed=$5

    python train_imagenet_symbolic.py \
            --model "resnet50_v1b" \
            --batch-size 128 \
            --num-epochs 42 \
            --warmup-epochs 5 \
            --lr 10 \
            --lr-step-epochs "pow2" \
            --lars-eta 0.001 \
            --lars-eps 0 \
            --log-interval 20 \
            --use-rec \
            --rec-train $DATA_DIR/train.rec \
            --rec-train-idx $DATA_DIR/train.idx \
            --rec-val $DATA_DIR/val.rec \
            --rec-val-idx $DATA_DIR/val.idx \
            --horovod \
            --random-seed $seed \
            --wd 0.0002 \
            --optimizer "sgdwfastlars" \
            --label-smoothing 0.1 \
            --momentum 0.9 \
            --num-examples 1281167 \
            --accuracy-target "0.759" \
            --use-symbolic

    popd
}

##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

$0: Execute a function by passing it as an argument to the script:

Possible commands:

EOF
    declare -F | cut -d' ' -f3
    echo
fi
