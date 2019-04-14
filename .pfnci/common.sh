#! /usr/bin/env sh
set -eux

if [ ${CHAINER} = stable ]; then
    CHAINER=5.3.0
elif [ ${CHAINER} = latest ]; then
    CHAINER=6.0.0b3
fi

if [ ${OPTIONAL_MODULES} -gt 0 ]; then
    DOCKER_IMAGE=hakuyume/chainercv:chainer${CHAINER}-devel
else
    DOCKER_IMAGE=hakuyume/chainercv:chainer${CHAINER}-devel-minimal
fi

fallocate -l 12G /swap
chmod 600 /swap
mkswap /swap
swapon -p 5 /swap

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=75%
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=75%
