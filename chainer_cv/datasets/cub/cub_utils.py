import os.path as osp
from skimage.io import imread
import subprocess

import chainer
from chainer.dataset import download

from chainer_cv import utils


root = 'yuyu2172/chainer-cv/cub'
url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'\
    'CUB_200_2011.tgz'


def get_cub():
    data_root = download.get_dataset_directory(root)
    base_path = osp.join(data_root, 'CUB_200_2011')
    if osp.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(url)

    subprocess.call(
        ['tar xzf {} -C {}'.format(download_file_path, data_root)], shell=True)
    return base_path


class CUBDatasetBase(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir='auto', crop_bbox=True):
        if data_dir == 'auto':
            data_dir = get_cub()
        self.data_dir = data_dir

        images_file = osp.join(data_dir, 'images.txt')
        bboxes_file = osp.join(data_dir, 'bounding_boxes.txt')

        self.fns = [fn.strip().split()[1] for fn in open(images_file)]
        bboxes = [bbox.split()[1:] for bbox in open(bboxes_file)]
        self.bboxes = [[int(float(elem)) for elem in bbox] for bbox in bboxes]

        self.crop_bbox = crop_bbox

    def __len__(self):
        return len(self.fns)

    def get_raw_data(self, i):
        img = imread(osp.join(self.data_dir, 'images', self.fns[i]))  # RGB

        if self.crop_bbox:
            bbox = self.bboxes[i]  # (x, y, width, height)
            img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
        return img
