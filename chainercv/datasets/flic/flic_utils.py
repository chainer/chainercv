import os

from chainer.dataset import download
from chainercv import utils

root = 'pfnet/chainercv/flic'

url = 'http://vision.grasp.upenn.edu/video/FLIC-full.zip'

flic_joint_names = [
    'lsho',
    'lelb',
    'lwri',
    'rsho',
    'relb',
    'rwri',
    'lhip',
    'lkne',
    'lank',
    'rhip',
    'rkne',
    'rank',
    'leye',
    'reye',
    'lear',
    'rear',
    'nose',
    'msho',
    'mhip',
    'mear',
    'mtorso',
    'mluarm',
    'mruarm',
    'mllarm',
    'mrlarm',
    'mluleg',
    'mruleg',
    'mllleg',
    'mrlleg'
]


def get_flic():
    data_root = download.get_dataset_directory(root)
    dataset_dir = os.path.join(data_root, 'FLIC-full')
    if not os.path.exists(dataset_dir):
        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, data_root, ext)

    return dataset_dir
