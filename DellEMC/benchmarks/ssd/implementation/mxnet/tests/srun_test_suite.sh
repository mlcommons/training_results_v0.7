#!/usr/bin/env bash

USER=`whoami`
: "${BASE_OUTPUT_DIR:=tests/results}"
: "${BASE_LOG_DIR:=tests/results}"
: "${CONTAINER_IMAGE:=nvcr.io/nvidian/swdl/mlperf_mxnet_ssd:master-devel}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                BASE_OUTPUT_DIR=$1
                                ;;
        -l | --log-dir )        shift
                                BASE_LOG_DIR=$1
                                ;;
        -d | --container )      shift
                                CONTAINER_IMAGE=$1
                                ;;
    esac
    shift
done


TESTS=(
    # Single Node configs
    "config_DGX1_01x08x56.sh 1"
    "config_DGX2_01x16x56.sh 1"
    "config_DGX2_01x16x56_amp.sh 10"
    "config_DGX2_01x16x56_fp32.sh 1"
    "config_DGX2_01x16x56_nchw.sh 1"
    "config_DGX2_01x16x56_unfuse_bn.sh 1"

    # 4 Nodes, Batch size=1536
    "config_DGX2_multi_04x16x24.sh 10"
    "config_DGX2_multi_04x16x24_pre02.sh 10"
    "config_DGX2_multi_04x16x24_pre04.sh 10"
    "config_DGX2_multi_04x16x24_pre08.sh 10"
    "config_DGX2_multi_04x16x24_pre16.sh 10"
    "config_DGX2_multi_04x16x24_amp.sh 10"

    # 4 Nodes, Batch size=2048
    "config_DGX2_multi_04x16x32.sh 10"
    "config_DGX2_multi_04x16x32_pre02.sh 10"
    "config_DGX2_multi_04x16x32_pre04.sh 10"
    "config_DGX2_multi_04x16x32_pre08.sh 10"
    "config_DGX2_multi_04x16x32_pre16.sh 10"
    "config_DGX2_multi_04x16x32_amp.sh 10"

    # 8 Nodes, Batch size=1536
    "config_DGX2_multi_08x16x12x02.sh 10"
    "config_DGX2_multi_08x16x12x02_pre02.sh 10"
    "config_DGX2_multi_08x16x12x02_pre04.sh 10"
    "config_DGX2_multi_08x16x12x02_pre08.sh 10"
    "config_DGX2_multi_08x16x12x02_pre16.sh 10"
    "config_DGX2_multi_08x16x12x02_amp.sh 10"

    # 8 Nodes, Batch size=2048
    "config_DGX2_multi_08x16x16.sh 10"
    "config_DGX2_multi_08x16x16_pre02.sh 10"
    "config_DGX2_multi_08x16x16_pre04.sh 10"
    "config_DGX2_multi_08x16x16_pre08.sh 10"
    "config_DGX2_multi_08x16x16_pre16.sh 10"
    "config_DGX2_multi_08x16x16_amp.sh 10"

    # 16 Nodes, Batch size=1536
    "config_DGX2_multi_16x16x06x04.sh 10"
    "config_DGX2_multi_16x16x06x04_pre02.sh 10"
    "config_DGX2_multi_16x16x06x04_pre04.sh 10"
    "config_DGX2_multi_16x16x06x04_pre08.sh 10"
    "config_DGX2_multi_16x16x06x04_pre16.sh 10"
    "config_DGX2_multi_16x16x06x04_amp.sh 10"

    # 16 Nodes, Batch size=2048
    "config_DGX2_multi_16x16x08x02.sh 10"
    "config_DGX2_multi_16x16x08x02_pre02.sh 10"
    "config_DGX2_multi_16x16x08x02_pre04.sh 10"
    "config_DGX2_multi_16x16x08x02_pre08.sh 10"
    "config_DGX2_multi_16x16x08x02_pre16.sh 10"
    "config_DGX2_multi_16x16x08x02_amp.sh 10"

    # 32 Nodes, Batch size=2048
    "config_DGX2_multi_32x16x04x04.sh 10"
    "config_DGX2_multi_32x16x04x04_pre02.sh 10"
    "config_DGX2_multi_32x16x04x04_pre04.sh 10"
    "config_DGX2_multi_32x16x04x04_pre08.sh 10"
    "config_DGX2_multi_32x16x04x04_pre16.sh 10"
    "config_DGX2_multi_32x16x04x04_pre32.sh 10"
    "config_DGX2_multi_32x16x04x04_amp.sh 10"

    # 64 Nodes, Batch size=2048
    "config_DGX2_multi_64x16x02x08.sh 10"
    "config_DGX2_multi_64x16x02x08_pre02.sh 10"
    "config_DGX2_multi_64x16x02x08_pre04.sh 10"
    "config_DGX2_multi_64x16x02x08_pre08.sh 10"
    "config_DGX2_multi_64x16x02x08_pre16.sh 10"
    "config_DGX2_multi_64x16x02x08_pre32.sh 10"
    "config_DGX2_multi_64x16x02x08_pre64.sh 10"
    "config_DGX2_multi_64x16x02x08_amp.sh 10"
    )

GIT_SHA=`git rev-parse HEAD`
GIT_SHA_SHORT=${GIT_SHA::5}
SUFFIX=`date +%s`
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${GIT_SHA_SHORT}_${SUFFIX}"
BASE_LOG_DIR="${BASE_LOG_DIR}/${GIT_SHA_SHORT}_${SUFFIX}"

echo "Using output dir: ${BASE_OUTPUT_DIR}"
echo "Log files dir: ${BASE_OUTPUT_DIR}"

for i in ${!TESTS[@]}; do
    test=${TESTS[$i]}
    config=$(echo $test | awk '{print $1}')
    n=$(echo $test | awk '{print $2}')
    config_name=${config#"config_"}
    config_name=${config_name%".sh"}
    log_dir="${BASE_LOG_DIR}/${config_name}"
    output_dir="${BASE_OUTPUT_DIR}/${config_name}"
    job_name=${config_name}_${GIT_SHA_SHORT}_${SUFFIX}
    tests/run_config.sh -n ${n} -l ${log_dir} -o ${output_dir} -d ${CONTAINER_IMAGE} -c ${config} -j ${job_name}
done
