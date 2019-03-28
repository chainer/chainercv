#! /usr/bin/env sh
set -eux

. .pfnci/common.sh

apt-get install -y --no-install-recommends unzip
gsutil -q cp gs://chainercv-pfn-public-ci/datasets-tiny.zip ${TEMP}/
unzip -q ${TEMP}/datasets-tiny.zip -d ${TEMP}/
rm ${TEMP}/datasets-tiny.zip

curl -L https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg \
     -o ${TEMP}/sample.jpg

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --volume ${TEMP}/sample.jpg:/sample.jpg \
       --env PYTHON=python${PYTHON} \
       --env MPIEXEC='mpiexec -n 2 --allow-run-as-root' \
       --env MPLBACKEND=agg \
       --env CHAINERCV_DOWNLOAD_REPORT=OFF \
       --env PFNCI_SKIP='echo SKIP:' \
       --env SAMPLE_IMAGE=/sample.jpg \
       ${DOCKER_IMAGE}
       sh -ex << EOD
pip${PYTHON} install --user -e .
for SCRIPT in \$(find examples_tests/ -type f -name '*.sh')
do
    sh -ex \${SCRIPT}
done
EOD
