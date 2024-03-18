# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM nvcr.io/nvidia/clara/bionemo-framework:1.2

# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENV PYTHON=python3
ENV PYTHON_VERSION=3.10.12
ENV PYTHON_SHORT_VERSION=3.10
ENV MAMBA_VERSION=23.11.0-0
ENV PYTORCH_VERSION=2.1.0
ENV DEBIAN_FRONTEND=noninteractive
ENV DLC_CONTAINER_TYPE=inference
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="${BIONEMO_HOME}:${PATH}"
ENV TQDM_POSITION=-1
ENV MODEL_PATH $BIONEMO_HOME/models

COPY serve .
COPY inference.py .
COPY wsgi.py .
COPY nginx.conf .

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean \
    && apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 --no-cache-dir install --upgrade pip \
    && pip --no-cache-dir install \
    boto3 \
    "sagemaker>=2,<3" \
    flask \
    gunicorn \
    gevent \
    ujson \
    && rm -rf /root/.cache | true \
    && rm "$HOME/.aws/config"

WORKDIR $BIONEMO_HOME
EXPOSE 8080
ENTRYPOINT ["/usr/bin/python"]
CMD ["serve"]