#! /usr/bin/env sh
set -eux

. .pfnci/common.sh

apt-get install -y --no-install-recommends unzip
gsutil -q cp gs://chainercv-pfn-public-ci/datasets-tiny.zip ${TEMP}/
unzip -q ${TEMP}/datasets-tiny.zip -d ${TEMP}/
rm ${TEMP}/datasets-tiny.zip

docker run --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
pip${PYTHON} install --user pytest-xdist
pip${PYTHON} install --user -e .
python${PYTHON} -m pytest --color=no -n $(nproc) \
                -m 'not pfnci_skip and not gpu and not mpi' tests/
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no \
        -m 'not pfnci_skip and not gpu and mpi' tests/
EOD
