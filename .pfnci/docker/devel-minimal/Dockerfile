FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    python-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://bootstrap.pypa.io/get-pip.py \
    && python2 get-pip.py --no-cache-dir \
    && python3 get-pip.py --no-cache-dir \
    && rm get-pip.py

RUN pip2 install --no-cache-dir \
    cython \
    mock \
    pytest
RUN pip3 install --no-cache-dir \
    cython \
    mock \
    pytest

ARG CHAINER
RUN [ -z ${CHAINER} ] || pip2 install --no-cache-dir \
    chainer==${CHAINER} \
    cupy-cuda92==${CHAINER}
RUN [ -z ${CHAINER} ] || pip3 install --no-cache-dir \
    chainer==${CHAINER} \
    cupy-cuda92==${CHAINER}
