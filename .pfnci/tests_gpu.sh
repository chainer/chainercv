#! /usr/bin/env sh
set -eux

. .pfnci/common.sh

mkdir -p ${TEMP}/.chainer

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
pip${PYTHON} install --user -e .
python${PYTHON} -m pytest --color=no \
                -m 'not pfnci_skip and gpu and not mpi' tests/
if which mpiexec; then
    mpiexec -n 2 --allow-run-as-root \
            python${PYTHON} -m pytest --color=no \
            -m 'not pfnci_skip and gpu and mpi' tests/
fi
EOD
