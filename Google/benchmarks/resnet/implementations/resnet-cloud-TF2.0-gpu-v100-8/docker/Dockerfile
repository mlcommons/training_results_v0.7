# Docker for MLPerf v0.7 Cloud GPU Resnet submission
# Based on tf-hightly-gpu (at 20200624) & the 20200620 pip packages

FROM ubuntu:18.04 as base

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get install -y --no-install-recommends dialog apt-utils

RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION=10.1.243

ENV CUDA_PKG_VERSION=10-1=10.1.243-1

RUN apt-get update && \
    apt-get install -y --no-install-recommends cuda-cudart-$CUDA_PKG_VERSION cuda-compat-10-1 && \
    ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES=all

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV NVIDIA_REQUIRE_CUDA=cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411

ARG ARCH

ARG CUDA=10.1

ARG CUDNN=7.6.4.38-1

ARG CUDNN_MAJOR_VERSION=7

ARG LIB_DIR_PREFIX=x86_64

ARG LIBNVINFER=6.0.1-1

ARG LIBNVINFER_MAJOR_VERSION=6

ARG LIB_DIR_PREFIX=x86_64

RUN ["/bin/bash", "-c", "apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas10=10.2.1.243-1 \
        cuda-nvrtc-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip"]

RUN ["/bin/bash", "-c", "[[ \"${ARCH}\" = \"ppc64le\" ]] || { \
        apt-get update && \
        apt-get install -y --no-install-recommends \
          libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
          libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; }"]

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN ["/bin/bash", "-c", "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
        echo \"/usr/local/cuda/lib64/stubs\" > /etc/ld.so.conf.d/z-cuda-stubs.conf && \
        ldconfig"]

ENV LANG=C.UTF-8

RUN ["/bin/bash", "-c", "apt-get update && apt-get install -y python3 python3-pip"]

RUN ["/bin/bash", "-c", "python3 -m pip --no-cache-dir install --upgrade pip setuptools"]

RUN ["/bin/bash", "-c", "ln -s $(which python3) /usr/local/bin/python"]

ARG TF_PACKAGE=tf-nightly-gpu

ARG TF_PACKAGE_VERSION=2.3.0.dev20200620

RUN ["/bin/bash", "-c", "python3 -m pip install --no-cache-dir \
        ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}} \
        tf-estimator-nightly==2.3.0.dev2020062001 \
        tb-nightly==2.3.0a20200620 \
        tensorflow-model-optimization==0.3.0"]

RUN ["/bin/bash", "-c", "apt-get update && apt-get install -y vim nano"]
