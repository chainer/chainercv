#! /usr/bin/env sh
set -eux

. $(dirname $0)/common.sh

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/root/ --workdir /root/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
. ./install.sh
cd chainercv/
python${PYTHON} -m pytest --color=no \
                -m 'not flexci_skip and gpu and not mpi' tests/
if which mpiexec; then
    mpiexec -n 2 --allow-run-as-root \
            python${PYTHON} -m pytest --color=no \
            -m 'not flexci_skip and gpu and mpi' tests/
fi
EOD
