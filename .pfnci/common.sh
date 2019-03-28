#! /usr/bin/env sh
set -eux

if [ ${CHAINER} = stable ]; then
    CHAINER=5.2.0
elif [ ${CHAINER} = latest ]; then
    CHAINER=6.0.0b2
fi

if [ ${OPTIONAL_MODULES} -gt 0 ]; then
    DOCKER_IMAGE=hakuyume/chainercv:chainer${CHAINER}-devel
else
    DOCKER_IMAGE=hakuyume/chainercv:chainer${CHAINER}-devel-minimal
fi

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%
