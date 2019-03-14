#! /usr/bin/env sh
set -eux

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%
mkdir -p ${TEMP}/.chainer

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ex << EOD
pip${PYTHON} install --user -e .
python${PYTHON} -m pytest --color=no \
                -m 'not pfnci_skip and gpu and not mpi' tests/
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no \
        -m 'not pfnci_skip and gpu and mpi' tests/
EOD
