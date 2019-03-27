#! /usr/bin/env sh
set -eux

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%
apt-get install -y --no-install-recommends unzip
gsutil -q cp gs://chainercv-pfn-public-ci/datasets-tiny.zip ${TEMP}/
unzip -q ${TEMP}/datasets-tiny.zip -d ${TEMP}/
rm ${TEMP}/datasets-tiny.zip

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env PYTHON=python${PYTHON} \
       --env MPIEXEC='mpiexec -n 2 --allow-run-as-root' \
       --env MPLBACKEND=agg \
       --env PFNCI_SKIP='echo SKIP:' \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ex << EOD
pip${PYTHON} install --user -e .
for SCRIPT in \$(find examples_tests/ -type f -name '*.sh')
do
    sh -ex \${SCRIPT}
done
EOD
