#! /usr/bin/env sh
set -eu

if [ "$(nvidia-smi --query-gpu=count --format=csv,noheader)" -gt 0 ]; then
    MARKS='gpu and not slow'
else
    MARKS='not gpu and not slow'
fi

docker run --runtime=nvidia --interactive --rm \
       --volume $(realpath .):/mnt --workdir /mnt \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -e << EOD
pip${PYTHON} install --user -e .
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m '${MARKS}' tests
EOD
