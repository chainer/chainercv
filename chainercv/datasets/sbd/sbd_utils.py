import filelock
import os
import six

from chainer.dataset import download

from chainercv.datasets.voc.voc_utils \
    import voc_instance_segmentation_label_names
from chainercv import utils

root = 'pfnet/chainercv/sbd'
url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA
train_voc2012_url = 'http://home.bharathh.info/pubs/codes/SBD/train_noval.txt'


def _generate_voc2012_txt(base_path):
    with open(os.path.join(base_path, 'train.txt'), 'r') as f:
        train_ids = f.read().split('\n')[:-1]
    with open(os.path.join(base_path, 'val.txt'), 'r') as f:
        val_ids = f.read().split('\n')[:-1]
    with open(os.path.join(base_path, 'train_voc2012.txt'), 'r') as f:
        train_voc2012_ids = f.read().split('\n')[:-1]
    all_ids = list(set(train_ids + val_ids))
    val_voc2012_ids = [i for i in all_ids if i not in train_voc2012_ids]

    with open(os.path.join(base_path, 'val_voc2012.txt'), 'w') as f:
        f.write('\n'.join(sorted(val_voc2012_ids)))
    with open(os.path.join(base_path, 'trainval_voc2012.txt'), 'w') as f:
        f.write('\n'.join(sorted(all_ids)))


def get_sbd():
    # To support ChainerMN, the target directory should be locked.
    with filelock.FileLock(os.path.join(download.get_dataset_directory(
            'pfnet/chainercv/.lock'), 'sbd.lock')):
        data_root = download.get_dataset_directory(root)
        base_path = os.path.join(data_root, 'benchmark_RELEASE/dataset')

        train_voc2012_file = os.path.join(base_path, 'train_voc2012.txt')
        if os.path.exists(train_voc2012_file):
            # skip downloading
            return base_path

        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, data_root, ext)

        six.moves.urllib.request.urlretrieve(
            train_voc2012_url, train_voc2012_file)
        _generate_voc2012_txt(base_path)

    return base_path


sbd_instance_segmentation_label_names = voc_instance_segmentation_label_names
