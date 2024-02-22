# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM nvcr.io/nvidia/clara/bionemo-framework:1.1 as common

ENV PYTHON=python3
ENV PYTHON_VERSION=3.10.12
ENV PYTHON_SHORT_VERSION=3.10
ENV MAMBA_VERSION=23.11.0-0
ENV PYTORCH_VERSION=2.1.0
# SageMaker Profiler Binary
ENV SMP_URL=https://smppy.s3.amazonaws.com/pytorch/cu121/smprof-0.3.334-cp310-cp310-linux_x86_64.whl
ENV EFA_PATH=/opt/amazon/efa
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# This arg required to stop docker build waiting for region configuration while installing tz data from ubuntu 20
ENV DEBIAN_FRONTEND=noninteractive

# older archs are not supported on CUDA12.1 like 3.7 for P2 instance
# 5.2 is G3 EC2 instance, 7.5 is G4*, 7.0 is p3*, 8.0 is P4*, 8.6 is G5* and 9.0 is P5*
ENV TORCH_CUDA_ARCH_LIST="5.2;7.0+PTX;7.5;8.0;8.6;9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDNN_VERSION=8.9.2.26
ENV EFA_VERSION=1.30.0
ENV GDRCOPY_VERSION=2.3.1

ENV OPEN_MPI_PATH=/opt/amazon/openmpi

ENV DGLBACKEND=pytorch
ENV MANUAL_BUILD=0
ENV RDMAV_FORK_SAFE=1
ENV DLC_CONTAINER_TYPE=training

#Set NCCL_ASYNC_ERROR_HANDLING=1 for PyTorch to timeout on NCCL errors https://github.com/pytorch/elastic/issues/136
ENV NCCL_ASYNC_ERROR_HANDLING=1

ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# For EFA, below flags are needed to install EFA on docker image
#  -n, --no-verify	Skip EFA device verification and test
#  -l, --skip-limit-conf	Skip EFA limit configuration
#  -k, --skip-kmod	Skip EFA kmod installation
RUN mkdir /tmp/efa \
    && cd /tmp/efa \
    && curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && apt-get update \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/efa \
    && rm -rf /tmp/aws-efa-installer-${EFA_VERSION}.tar.gz \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="$OPEN_MPI_PATH/bin:$EFA_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH=$OPEN_MPI_PATH/lib/:$EFA_PATH/lib/:$LD_LIBRARY_PATH

RUN pip install --upgrade pip --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Selectively enable PyTorch Verbose Logging
ENV PYTORCH_API_USAGE_STDERR=1
ENV TORCH_LOGS=+dynamo,+aot,+inductor

WORKDIR /

# Install SageMaker Profiler Binary
RUN pip install --no-cache-dir -U ${SMP_URL}

# Install aws-ofi-nccl
RUN wget -O /tmp/ofi-aws.tar.gz https://github.com/aws/aws-ofi-nccl/releases/download/v1.8.0-aws/aws-ofi-nccl-1.8.0-aws.tar.gz \
    && tar -xvzf /tmp/ofi-aws.tar.gz -C /usr/local/bin \
    && rm /tmp/ofi-aws.tar.gz

# install PyTorch related python packages (depends on PyTorch)
RUN pip install --no-cache-dir \
    # needed by fastai.distributed
    # pin accelerate to 0.22.0 as newer version gives error "0.23.0 requires huggingface-hub, which is not installed."
    "accelerate==0.22.0" \
    fastai  \
    pillow \
    opencv-python \
    "requests>=2.31.0" \
    && pip uninstall -y dataclasses \
    && pip install --no-cache-dir torchtnt

# makes AllToAll complete successfully. Update will be included in NCCL 2.20.*
ENV NCCL_CUMEM_ENABLE=0
# preload to system nccl for PyTorch to use its statically linked NCCL
# ENV LD_PRELOAD="${CUDA_HOME}/lib/libnccl-net.so"

# Copy workaround script for incorrect hostname
COPY changehostname.c /
COPY start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh

RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

WORKDIR /root

# install SM packages
RUN pip install --no-cache-dir -U \
    smclarify \
    "sagemaker>=2,<3" \
    sagemaker-pytorch-training \
    sagemaker-training \
    numba \
    pandas \
    shap \
    scikit-learn

WORKDIR /

# Removing the cache as it is needed for security verification
RUN rm -rf /root/.cache | true \
    && rm "$HOME/.aws/config"

ENV TQDM_POSITION=-1
ENV MODEL_PATH $BIONEMO_HOME/models

ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]