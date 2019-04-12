#! /usr/bin/env sh
set -eux

STABLE=v5.3.0
LATEST=v6.0.0b3

fallocate -l 12G /swap
chmod 600 /swap
mkswap /swap
swapon -p 5 /swap

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=75%
gsutil -q cp gs://tmp-pfn-public-ci/chainercv/docker.tar - | tar -xf - -C /var/lib/docker/ || true
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=75%

docker_build() {
    if [ ${CHAINER} = stable ]; then
        CHAINER=${STABLE}
        CUPY=${STABLE}
    elif [ ${CHAINER} = latest ]; then
        CHAINER=${LATEST}
        CUPY=${LATEST}
    elif [ ${CHAINER} = master ]; then
        CHAINER=${CHAINER_MASTER}
        CUPY=${CUPY_MASTER}
    fi

    if [ ${OPTIONAL_MODULES} -gt 0 ]; then
        DOCKER_CONTEXT=.pfnci/docker/devel/
    else
        DOCKER_CONTEXT=.pfnci/docker/devel-minimal/
    fi

    docker build \
               --build-arg CHAINER=${CHAINER} \
               --build-arg CUPY=${CUPY} \
               "$@" \
               ${DOCKER_CONTEXT}
}
