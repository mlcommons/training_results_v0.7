#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_CUDA_VERSION=10
_DEFAULT_CUDNN_VERSION=7

# Run the TensorFlow configuration script, setting reasonable values for most
# of the options.
echo "Configuring tensorflow"

# //cc/tensorflow:build needs to know whether the user wants TensorRT support,
# so it can build extra libraries.
#if [ -z "${TF_NEED_TENSORRT}" ]; then
#  read -p "Enable TensorRT support? [y/N]: " yn
#  case $yn in
#      [Yy]* ) export TF_NEED_TENSORRT=1;;
#      * ) export TF_NEED_TENSORRT=0;;
#  esac
#fi

# //org_tensorflow//:configure script must be run from the TensorFlow repository
# root. Build the script in order to pull the repository contents from GitHub.
# The `bazel fetch` and `bazel sync` commands that are usually used to fetch
# external Bazel dependencies don't work correctly on the TensorFlow repository.
bazel --bazelrc=/dev/null build @org_tensorflow//:configure

# Get the Minigo workspace root.
workspace=$(bazel info workspace)

# External dependencies are stored in a directory named after the workspace
# directory name.
pushd bazel-`basename "${workspace}"`/external/org_tensorflow

CC_OPT_FLAGS="${cc_opt_flags}" \
PYTHON_BIN_PATH=${PYTHON_BIN_PATH} \
USE_DEFAULT_PYTHON_LIB_PATH="${USE_DEFAULT_PYTHON_LIB_PATH:-1}" \
TF_NEED_JEMALLOC=${TF_NEED_JEMALLOC:-0} \
TF_NEED_GCP=${TF_NEED_GCP:-0} \
TF_NEED_HDFS=${TF_NEED_HDFS:-0} \
TF_NEED_S3=${TF_NEED_S3:-0} \
TF_NEED_KAFKA=${TF_NEED_KAFKA:-0} \
TF_NEED_CUDA=${TF_NEED_CUDA:-0} \
TF_NEED_GDR=${TF_NEED_GDR:-0} \
TF_NEED_VERBS=${TF_NEED_VERBS:-0} \
TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL:-0} \
TF_NEED_ROCM=${TF_NEED_ROCM:-0} \
TF_CUDA_CLANG=${TF_CUDA_CLANG:-0} \
TF_DOWNLOAD_CLANG=${TF_DOWNLOAD_CLANG:-0} \
TF_NEED_TENSORRT=${TF_NEED_TENSORRT:-0} \
TF_NEED_MPI=${TF_NEED_MPI:-0} \
TF_SET_ANDROID_WORKSPACE=${TF_SET_ANDROID_WORKSPACE:-0} \
TF_NCCL_VERSION=${TF_NCCL_VERSION:-1.3} \
TF_ENABLE_XLA=${TF_ENABLE_XLA:-0} \
bazel --bazelrc=/dev/null run @org_tensorflow//:configure

. ${script_dir}/../set_avx2_build
BAZEL_OPTS="-c opt --config=mkl \
            --action_env=PATH \
            --action_env=LD_LIBRARY_PATH \
            $BAZEL_BUILD_OPTS \
            --copt=-DINTEL_MKLDNN"

#BAZEL_OPTS="-c opt \
#            --action_env=PATH \
#            --action_env=LD_LIBRARY_PATH"

# Copy from the TensorFlow output_base.
output_base=$(bazel info output_base)
popd

# Copy TensorFlow's bazelrc files to workspace.
cp ${output_base}/external/org_tensorflow/.bazelrc ${workspace}/tensorflow.bazelrc
cp ${output_base}/external/org_tensorflow/.tf_configure.bazelrc ${workspace}/tf_configure.bazelrc

echo "Building tensorflow package"
bazel run $BAZEL_OPTS \
  --copt=-Wno-comment \
  --copt=-Wno-deprecated-declarations \
  --copt=-Wno-ignored-attributes \
  --copt=-Wno-maybe-uninitialized \
  --copt=-Wno-sign-compare \
  --define=need_trt="$TF_NEED_TENSORRT" \
  //cc/tensorflow:build -- ${workspace}/cc/tensorflow

src_dir="bazel-bin/_solib_k8/_U\@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib"
dest_dir="lib"

if [ ! -d ${dest_dir} ]; then
    mkdir ${dest_dir}
fi

cp ${src_dir}/* ${dest_dir}
