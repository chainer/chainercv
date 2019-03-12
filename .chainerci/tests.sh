#! /usr/bin/env sh
set -eux

TEMP=$(mktemp -d)
mount -t tmpfs tmpfs ${TEMP}/ -o size=100%
apt-get install -y --no-install-recommends unzip
gsutil -q cp gs://chainercv-pfn-public-ci/datasets-tiny.zip ${TEMP}/
unzip -q ${TEMP}/datasets-tiny.zip -d ${TEMP}/
rm ${TEMP}/datasets-tiny.zip

set +x
# rename tests for pytest-xdist
for TEST in $(find tests/ -name '*.py')
do
    cat - << 'EOD' >> ${TEST}
def rename_tests(module_name):
    import inspect
    import re
    import sys
    import unittest

    module = sys.modules[module_name]
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(cls, unittest.TestCase):
            continue
        delattr(module, name)
        new_name = re.sub(r'(?s)_{.+}', '', name)
        new_cls = type(new_name, (cls,), {})
        setattr(module, new_name, new_cls)


rename_tests(__name__)
EOD
done
set -x

docker run --interactive --rm \
       --volume $(pwd):/chainercv/ --workdir /chainercv/ \
       --volume ${TEMP}/.chainer/:/root/.chainer/ \
       --env MPLBACKEND=agg \
       hakuyume/chainercv:chainer${CHAINER}-devel \
       sh -ex << EOD
pip${PYTHON} install --user pytest-xdist
pip${PYTHON} install --user -e .
python${PYTHON} -m pytest \
                --color=no -n $(nproc) -m 'not gpu and not mpi' tests/
mpiexec -n 2 --allow-run-as-root \
        python${PYTHON} -m pytest --color=no -m 'not gpu and mpi' tests/
EOD
