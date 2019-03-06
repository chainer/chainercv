#! /usr/bin/env sh
set -eu

docker run --runtime=nvidia --rm --volume $(realpath .):/mnt \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ec "pip${PYTHON} install --user -e /mnt; mpiexec -n 2 --allow-run-as-root python${PYTHON} -m pytest -m 'not slow' /mnt/tests"
