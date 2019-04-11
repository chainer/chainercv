FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    make \
    openssh-client \
    python-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://bootstrap.pypa.io/get-pip.py \
    && python2 get-pip.py --no-cache-dir \
    && python3 get-pip.py --no-cache-dir \
    && rm get-pip.py

RUN pip2 install --no-cache-dir numpy \
    && pip3 install --no-cache-dir numpy

ENV OPENCV_VERSION=3.4.1
RUN cd $(mktemp -d) \
    && curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz | tar zxf - \
    && mkdir opencv-${OPENCV_VERSION}/build \
    && cd opencv-${OPENCV_VERSION}/build \
    && cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=OFF \
    -DBUILD_WITH_DEBUG_INFO=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    && make -j $(nproc) \
    && make install \
    && cd - \
    && rm -rf $(pwd)

ENV OPENMPI_VERSION=3.0.1
RUN cd $(mktemp -d) \
    && curl -L https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSION%.*}/downloads/openmpi-${OPENMPI_VERSION}.tar.bz2 | tar jxf - \
    && cd openmpi-${OPENMPI_VERSION} \
    && ./configure --with-cuda \
    && make -j $(nproc) \
    && make install \
    && cd - \
    && rm -rf $(pwd)

RUN echo /usr/local/lib/ > /etc/ld.so.conf.d/local.conf && ldconfig

RUN pip2 install --no-cache-dir \
    cython \
    matplotlib==2 \
    mock \
    mpi4py \
    'git+https://github.com/cocodataset/coco.git#egg=pycocotools&subdirectory=PythonAPI' \
    pytest \
    scipy
RUN pip3 install --no-cache-dir \
    cython \
    matplotlib \
    mock \
    mpi4py \
    'git+https://github.com/cocodataset/coco.git#egg=pycocotools&subdirectory=PythonAPI' \
    pytest \
    scipy

ARG CHAINER
RUN [ -z ${CHAINER} ] || pip2 install --no-cache-dir \
    chainer==${CHAINER} \
    cupy-cuda92==${CHAINER}
RUN [ -z ${CHAINER} ] || pip3 install --no-cache-dir \
    chainer==${CHAINER} \
    cupy-cuda92==${CHAINER}
