#! /usr/bin/env sh
set -eux

STABLE=6.3.0
LATEST=7.0.0b3

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=100%
systemctl start docker.service
gcloud auth configure-docker

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%

REPOSITORY=${REPOSITORY:-chainercv}
case ${REPOSITORY} in
    chainercv)
        CUPY=${CHAINER}
        cp -a . ${TEMP}/chainercv
        ;;
    chainer)
        CHAINER=local
        CUPY=master
        cp -a . ${TEMP}/chainer
        mv ${TEMP}/chainer/chainercv/ ${TEMP}/
        ;;
    cupy)
        CHAINER=master
        CUPY=local
        cp -a . ${TEMP}/cupy
        mv ${TEMP}/cupy/chainercv/ ${TEMP}/
        ;;
esac
cd ${TEMP}/

case ${CHAINER} in
    stable)
        echo pip${PYTHON} install chainer==${STABLE} >> install.sh
        ;;
    latest)
        echo pip${PYTHON} install chainer==${LATEST} >> install.sh
        ;;
    master)
        CHAINER_MASTER=$(git ls-remote https://github.com/chainer/chainer.git master | cut -f1)
        echo pip${PYTHON} install \
             git+https://github.com/chainer/chainer.git@${CHAINER_MASTER}#egg=chainer >> install.sh
        ;;
    local)
        echo pip${PYTHON} install -e chainer/ >> install.sh
        ;;
esac

case ${CUPY} in
    stable)
        echo pip${PYTHON} install cupy-cuda92==${STABLE} >> install.sh
        ;;
    latest)
        echo pip${PYTHON} install cupy-cuda92==${LATEST} >> install.sh
        ;;
    master)
        CUPY_MASTER=$(gsutil -q cp gs://tmp-asia-pfn-public-ci/cupy/wheel/master -)
        gsutil -q cp gs://tmp-asia-pfn-public-ci/cupy/wheel/${CUPY_MASTER}/cuda9.2/*.whl .
        echo pip${PYTHON} install cupy-*-cp${PYTHON}*-cp${PYTHON}*-linux_x86_64.whl >> install.sh
        ;;
    local)
        echo pip${PYTHON} install -e cupy/ >> install.sh
        ;;
esac

echo pip${PYTHON} install -e chainercv/ >> install.sh

if [ ${OPTIONAL_MODULES} -gt 0 ]; then
    DOCKER_TAG=devel
else
    DOCKER_TAG=devel-minimal
fi
DOCKER_IMAGE=asia.gcr.io/pfn-public-ci/chainercv:${DOCKER_TAG}
docker pull ${DOCKER_IMAGE} || true
docker build \
       --cache-from ${DOCKER_IMAGE} \
       --tag ${DOCKER_IMAGE} \
       chainercv/.pfnci/docker/${DOCKER_TAG}/
