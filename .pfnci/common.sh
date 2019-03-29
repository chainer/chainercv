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

mkfs.btrfs /dev/nvme0n1
mount /dev/nvme0n1 /mnt/
btrfs subvolume create /mnt/docker
btrfs subvolume create /mnt/temp
umount /mnt/

systemctl stop docker.service
mount /dev/nvme0n1 /var/lib/docker/ -o subvol=docker
systemctl start docker.service

TEMP=$(mktemp -d)
mount /dev/nvme0n1 ${TEMP}/ -o subvol=temp
