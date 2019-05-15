#! /usr/bin/env sh
set -eux

STABLE=5.4.0
# 6.0.0rc1 does not work
# https://github.com/chainer/chainer/issues/6825
LATEST=6.0.0b3

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=100%
gsutil -q cp gs://tmp-pfn-public-ci/chainercv/docker.tar - | tar -xf - -C /var/lib/docker/ || true
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%

REPOSITORY=${REPOSITORY:-chainercv}
if [ ${REPOSITORY} = chainercv ]; then
    cp -a . ${TEMP}/chainercv
elif [ ${REPOSITORY} = chainer ]; then
    CHAINER=local
    cp -a . ${TEMP}/chainer
    mv ${TEMP}/chainer/chainercv/ ${TEMP}/
fi
cd ${TEMP}/

if [ ${CHAINER} = stable ]; then
    echo pip${PYTHON} install \
         chainer==${STABLE} \
         cupy-cuda92==${STABLE} >> install.sh
elif [ ${CHAINER} = latest ]; then
    echo pip${PYTHON} install \
         chainer==${LATEST} \
         cupy-cuda92==${LATEST} >> install.sh
elif [ ${CHAINER} = master ]; then
    CHAINER_MASTER=$(git ls-remote https://github.com/chainer/chainer.git master | cut -f1)
    echo pip${PYTHON} install \
         git+https://github.com/chainer/chainer.git@${CHAINER_MASTER}#egg=chainer >> install.sh

    CUPY_MASTER=$(gsutil -q cp gs://tmp-pfn-public-ci/cupy/wheel/master -)
    gsutil -q cp gs://tmp-pfn-public-ci/cupy/wheel/${CUPY_MASTER}/cuda9.2/*.whl .
    echo pip${PYTHON} install cupy-*-cp${PYTHON}*-cp${PYTHON}*-linux_x86_64.whl >> install.sh
elif [ ${CHAINER} = local ]; then
    echo pip${PYTHON} install -e chainer/ >> install.sh

    CUPY_MASTER=$(gsutil -q cp gs://tmp-pfn-public-ci/cupy/wheel/master -)
    gsutil -q cp gs://tmp-pfn-public-ci/cupy/wheel/${CUPY_MASTER}/cuda9.2/*.whl .
    echo pip${PYTHON} install cupy-*-cp${PYTHON}*-cp${PYTHON}*-linux_x86_64.whl >> install.sh
fi

echo pip${PYTHON} install -e chainercv/ >> install.sh

if [ ${OPTIONAL_MODULES} -gt 0 ]; then
    DOCKER_IMAGE=devel
else
    DOCKER_IMAGE=devel-minimal
fi
docker build -t ${DOCKER_IMAGE} chainercv/.pfnci/docker/${DOCKER_IMAGE}/
