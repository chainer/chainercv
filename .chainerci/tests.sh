#! /usr/bin/env sh
set -eu

docker run --runtime=nvidia -i --rm \
       --volume $(realpath .):/mnt --workdir /mnt \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -e << EOD
pip${PYTHON} install --user -e .
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m 'not slow' tests
EOD
