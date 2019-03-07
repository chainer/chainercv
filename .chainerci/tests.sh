#! /usr/bin/env sh
set -eux

if [ "$(nvidia-smi --query-gpu=count --format=csv,noheader)" -gt 0 ]; then
    RUNTIME='--runtime=nvidia'
    MARKS='gpu and not slow'
else
    RUNTIME=
    MARKS='not gpu and not slow'
fi

docker run ${RUNTIME} --interactive --rm \
       --volume $(realpath .):/mnt --workdir /mnt \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -e << EOD
pip${PYTHON} install --user -e .
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m '${MARKS}' tests
EOD
