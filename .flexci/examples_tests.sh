#! /usr/bin/env sh
set -eux

. $(dirname $0)/common.sh

for ZIP in datasets-tiny.zip models.zip
do
    gsutil -q cp gs://chainercv-asia-pfn-public-ci/${ZIP} .
    unzip -q ${ZIP}
    rm ${ZIP}
done

curl -L https://cloud.githubusercontent.com/assets/2062128/26187667/9cb236da-3bd5-11e7-8bcf-7dbd4302e2dc.jpg \
     -o sample.jpg

docker run --runtime=nvidia --interactive --rm \
       --volume $(pwd):/root/ --workdir /root/ \
       --env SAMPLE_IMAGE=/root/sample.jpg \
       --env PYTHON=python${PYTHON} \
       --env MPIEXEC='mpiexec -n 2 --allow-run-as-root' \
       --env MPLBACKEND=agg \
       --env CHAINERCV_DOWNLOAD_REPORT=OFF \
       --env FLEXCI_SKIP='echo SKIP:' \
       ${DOCKER_IMAGE} \
       sh -ex << EOD
. ./install.sh
cd chainercv/
for SCRIPT in \$(find examples_tests/ -type f -name '*.sh')
do
    sh -ex \${SCRIPT}
done
EOD
