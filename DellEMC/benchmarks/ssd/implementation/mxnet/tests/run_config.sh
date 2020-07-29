#!/usr/bin/env bash

USER=`whoami`
: "${NEXP:=1}"
: "${CONFIG:=config_DGX1.sh}"
: "${JOB_NAME:=job}"
: "${CONTAINER_IMAGE:=nvcr.io/nvidian/swdl/mlperf_mxnet_ssd:master-devel}"

while [ "$1" != "" ]; do
    case $1 in
        -n | --num-runs )       shift
                                NEXP=$1
                                ;;
        -c | --config )         shift
                                CONFIG=$1
                                ;;
        -o | --output-dir )     shift
                                BASE_OUTPUT_DIR=$1
                                ;;
        -l | --log-dir )        shift
                                BASE_LOG_DIR=$1
                                ;;
        -d | --container )      shift
                                CONTAINER_IMAGE=$1
                                ;;
        -j | --job-name )       shift
                                JOB_NAME=$1
                                ;;
    esac
    shift
done

source ${CONFIG}
COCO_FOLDER=/raid/datasets/coco/coco-2017/coco2017
BACKBONE_FOLDER=/gpfs/fs1/akiswani/workspace/ssd/ssd-backbone
CONFIG_NAME=${CONFIG#"config_"}
CONFIG_NAME=${CONFIG_NAME%".sh"}

mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p ${BASE_LOG_DIR}
# Run experiments
for i in $(seq 1 "${NEXP}"); do
    echo "[${i}/${NEXP}] Running config ${CONFIG_NAME}"
    mkdir -p ${BASE_OUTPUT_DIR}/${i}
    # Run experiment
    nohup srun --mpi=pmix \
         --account=mlperft-ssd \
         --job-name="${JOB_NAME}_${i}" \
         --container-image="${CONTAINER_IMAGE}" \
         --container-workdir=/ssd \
         --container-mounts=$(pwd):/ssd,${COCO_FOLDER}:/data/coco2017,${BACKBONE_FOLDER}:/pretrained/mxnet,${BASE_OUTPUT_DIR}/${i}:/results \
         --nodes="${DGXNNODES}" \
         --ntasks-per-node="${DGXNGPU}" \
         --time="${WALLTIME}" \
        ./run_and_time.sh < /dev/null > ${BASE_LOG_DIR}/job_${i}.out 2>&1 &
done
echo ""
