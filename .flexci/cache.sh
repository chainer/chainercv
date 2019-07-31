#! /usr/bin/env sh
set -eux

systemctl stop docker.service
mount -t tmpfs tmpfs /var/lib/docker/ -o size=100%
systemctl start docker.service
gcloud auth configure-docker

for DOCKER_TAG in devel devel-minimal
do
    DOCKER_IMAGE=asia.gcr.io/pfn-public-ci/chainercv:${DOCKER_TAG}
    docker pull ${DOCKER_IMAGE} || true
    docker build \
       --cache-from ${DOCKER_IMAGE} \
       --tag ${DOCKER_IMAGE} \
       .flexci/docker/${DOCKER_TAG}/
    docker push ${DOCKER_IMAGE}
done
