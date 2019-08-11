#! /usr/bin/env sh
set -eux

. $(dirname $0)/common.sh

for ZIP in datasets-tiny.zip models.zip
do
    gsutil -q cp gs://chainercv-asia-pfn-public-ci/${ZIP} .
    unzip -q ${ZIP}
    rm ${ZIP}
done

gsutil -q cp gs://chainercv-pfn-public-ci/.coveralls.yml chainercv/

docker run --interactive --rm \
       --volume $(pwd):/root/ --workdir /root/ \
       --env MPLBACKEND=agg \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
. ./install.sh
pip${PYTHON} install --user pytest-xdist
cd chainercv/
python${PYTHON} -m pytest --color=no -n $(nproc) \
                --cov=chainercv/ --cov-report= \
                -m 'not pfnci_skip and not gpu and not mpi' tests/
if which mpiexec; then
    mpiexec -n 2 --allow-run-as-root \
            python${PYTHON} -m pytest --color=no \
            --cov=chainercv/ --cov-report= --cov-append \
            -m 'not pfnci_skip and not gpu and mpi' tests/
fi
coveralls
EOD
