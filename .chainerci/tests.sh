#! /usr/bin/env sh
set -eux

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%

if nvidia-smi --query-gpu=count --format=csv; then
    RUNTIME='--runtime=nvidia'
    NUM=
    MARKS='gpu'

    mkdir -p ${TEMP}/.chainer
else
    RUNTIME=
    NUM="-n $(nproc)"
    MARKS='not gpu'

    apt-get install -y --no-install-recommends unzip
    gsutil cp gs://chainercv-pfn-public-ci/datasets-tiny.zip ${TEMP}/
    unzip -q ${TEMP}/datasets-tiny.zip -d ${TEMP}/
    rm ${TEMP}/datasets-tiny.zip
fi

docker run ${RUNTIME} --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ex << EOD
pip${PYTHON} install --user pytest-xdist
pip${PYTHON} install --user -e .
python${PYTHON} -m pytest --color=no ${NUM} -m '${MARKS} and not mpi' tests/
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m '${MARKS} and mpi' tests/
EOD
