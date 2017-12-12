import os

from chainer.dataset import download
from chainercv import utils

root = 'pfnet/chainercv/flic'

urls = [
    'http://vision.grasp.upenn.edu/video/FLIC-full.zip',

    'http://cims.nyu.edu/~tompson/data/tr_plus_indices.mat',

]

flic_joint_label_names = [
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

    if not os.path.exists(os.path.join(data_root,
                                       'FLIC-full')):
        download_file_path = utils.cached_download(urls[0])
        ext = os.path.splitext(urls[0])[1]
        utils.extractall(download_file_path,
                         data_root,
                         ext)

    return data_root
