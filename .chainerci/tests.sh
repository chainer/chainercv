#! /usr/bin/env sh
set -eux

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%

if nvidia-smi --query-gpu=count --format=csv; then
    mkdir -p ${TEMP}/.chainer

    docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ex << EOD
pip${PYTHON} install --user -e .
python${PYTHON} -m pytest --color=no -m 'gpu and not mpi' tests/
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m 'gpu and mpi' tests/
EOD
else
    apt-get install -y --no-install-recommends unzip
    gsutil cp gs://chainercv-pfn-public-ci/datasets-tiny.zip ${TEMP}/
    unzip -q ${TEMP}/datasets-tiny.zip -d ${TEMP}/
    rm ${TEMP}/datasets-tiny.zip

    docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ex << EOD
pip${PYTHON} install --user -e .
curl -LO https://raw.githubusercontent.com/imos/public/master/parallel-pytest.py
python${PYTHON} parallel-pytest.py \
                --pytest='python${PYTHON} -m pytest --color=no -m \'not gpu and not mpi\'' \
                --threads=$(nproc) \
                --directory=tests/
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m 'not gpu and mpi' tests/
EOD
fi
