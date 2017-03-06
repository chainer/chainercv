import os

import chainer
from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/cub'
url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'\
    'CUB_200_2011.tgz'


def get_cub():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'CUB_200_2011')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)
    return base_path


class CUBDatasetBase(chainer.dataset.DatasetMixin):

    """Base class for CUB dataset.

    """

    def __init__(self, data_dir='auto', crop_bbox=True):
        if data_dir == 'auto':
            data_dir = get_cub()
        self.data_dir = data_dir

        images_file = os.path.join(data_dir, 'images.txt')
        bboxes_file = os.path.join(data_dir, 'bounding_boxes.txt')

        self.fns = [fn.strip().split()[1] for fn in open(images_file)]
        bboxes = [bbox.split()[1:] for bbox in open(bboxes_file)]
        self.bboxes = [[int(float(elem)) for elem in bbox] for bbox in bboxes]

        self.crop_bbox = crop_bbox

    def __len__(self):
        return len(self.fns)
