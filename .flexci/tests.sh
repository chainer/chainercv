#! /usr/bin/env sh
set -eux

. $(dirname $0)/common.sh

for ZIP in datasets-tiny.zip models.zip
do
    gsutil -q cp gs://chainercv-asia-pfn-public-ci/${ZIP} .
    unzip -q ${ZIP}
    rm ${ZIP}
done

docker run --interactive --rm \
       --volume $(pwd):/root/ --workdir /root/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
. ./install.sh
pip${PYTHON} install --user pytest-xdist
cd chainercv/
python${PYTHON} -m pytest --color=no -n $(nproc) \
                -m 'not flexci_skip and not gpu and not mpi' tests/
if which mpiexec; then
    mpiexec -n 2 --allow-run-as-root \
            python${PYTHON} -m pytest --color=no \
            -m 'not flexci_skip and not gpu and mpi' tests/
fi
EOD
