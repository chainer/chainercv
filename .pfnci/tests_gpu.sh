#! /usr/bin/env sh
set -eux

. $(dirname $0)/common.sh

gsutil -q cp gs://chainercv-pfn-public-ci/.coveralls.yml .

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/root/ --workdir /root/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
. ./install.sh
cd chainercv/
python${PYTHON} -m pytest --color=no \
                --cov=chainercv/ --cov-report= \
                -m 'not pfnci_skip and gpu and not mpi' tests/
if which mpiexec; then
    mpiexec -n 2 --allow-run-as-root \
            python${PYTHON} -m pytest --color=no \
            --cov=chainercv/ --cov-report= --cov-append \
            -m 'not pfnci_skip and gpu and mpi' tests/
fi
coveralls
EOD
