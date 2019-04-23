#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/
gsutil -q cp gs://tmp-pfn-public-ci/chainercv/docker.tar - | tar -xf - -C /var/lib/docker/ || true
systemctl start docker.service

docker build -t devel .pfnci/docker/devel/
docker build -t devel-minimal .pfnci/docker/devel-minimal/

docker system prune --force
systemctl stop docker.service
tar -cf - -C /var/lib/docker/ -R . | gsutil -q cp - gs://tmp-pfn-public-ci/chainercv/docker.tar
