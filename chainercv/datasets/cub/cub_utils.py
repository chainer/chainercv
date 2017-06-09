import numpy as np
import os

import chainer
from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/cub'
url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'\
    'CUB_200_2011.tgz'
mask_url = 'http://www.vision.caltech.edu/visipedia-data/'\
    'CUB-200-2011/segmentations.tgz'


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


def get_cub_mask():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'segmentations')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    download_file_path_mask = utils.cached_download(mask_url)
    ext_mask = os.path.splitext(mask_url)[1]
    utils.extractall(
        download_file_path_mask, data_root, ext_mask)
    return base_path


class CUBDatasetBase(chainer.dataset.DatasetMixin):

    """Base class for CUB dataset.

    """

    def __init__(self, data_dir='auto', mask_dir='auto', crop_bbox=True):
        if data_dir == 'auto':
            data_dir = get_cub()
        if mask_dir == 'auto':
            mask_dir = get_cub_mask()
        self.data_dir = data_dir
        self.mask_dir = mask_dir

        imgs_file = os.path.join(data_dir, 'images.txt')
        bboxes_file = os.path.join(data_dir, 'bounding_boxes.txt')

        self.filenames = [
            line.strip().split()[1] for line in open(imgs_file)]

        # (x_min, y_min, width, height)
        bboxes = np.array([
            tuple(map(float, line.split()[1:5]))
            for line in open(bboxes_file)])
        # (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
        bboxes[:, 2:] += bboxes[:, :2]
        # (x_min, y_min, width, height) -> (y_min, x_min, y_max, x_max)
        bboxes[:] = bboxes[:, [1, 0, 3, 2]]
        self.bboxes = bboxes.astype(np.float32)

        self.crop_bbox = crop_bbox

    def __len__(self):
        return len(self.filenames)
