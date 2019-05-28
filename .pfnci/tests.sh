#! /usr/bin/env sh
set -eux

. $(dirname $0)/common.sh

apt-get install -y --no-install-recommends unzip
gsutil -q cp gs://chainercv-pfn-public-ci/datasets-tiny.zip .
unzip -q datasets-tiny.zip
rm datasets-tiny.zip

docker run --interactive --rm \
       --volume $(pwd):/root/ --workdir /root/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
. ./install.sh
pip${PYTHON} install --user pytest-xdist
cd chainercv/
python${PYTHON} -m pytest --color=no -n $(nproc) \
                -m 'not pfnci_skip and not gpu and not mpi' tests/
if which mpiexec; then
    mpiexec -n 2 --allow-run-as-root \
            python${PYTHON} -m pytest --color=no \
            -m 'not pfnci_skip and not gpu and mpi' tests/
fi
EOD
