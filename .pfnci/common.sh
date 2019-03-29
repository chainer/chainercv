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

dd if=/dev/zero of=/swap bs=1G count=24
chmod 600 /swap
mkswap /swap
swapon -p 5 /swap

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=75%
systemctl start docker.service

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=75%
