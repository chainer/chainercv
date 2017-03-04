import numpy as np
import os
import pickle
from skimage.io import imread

import chainer
from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/unite_the_people'
urls = [
    'http://files.is.tue.mpg.de/classner/up/datasets/up-3d.zip',
    'http://files.is.tue.mpg.de/classner/up/datasets/up-p91.zip',
    'http://files.is.tue.mpg.de/classner/up/datasets/upi-s1h.zip'
]


def get_unite_the_people():
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'up-3d')
    if os.path.exists(base_path):
        # skip downloading
        return base_path

    for url in urls:
        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, data_root, ext)
    return base_path


class UniteThePeopleDataset(chainer.dataset.DatasetMixin):

    """Dataset class to load data from `Unite the People Dataset`_.

    .. _`Unite the People Dataset`:
        <https://arxiv.org/abs/1701.02468>

    Joint is a (3, 14) numpy.ndarray. The first axis represents
    (x cord, y coord, visible) of joints.

    """

    def __init__(self, data_dir='auto', mode='trainval'):
        if data_dir == 'auto':
            data_dir = get_unite_the_people()
        self.data_dir = data_dir

        if mode not in ['train', 'trainval', 'val', 'test']:
            raise ValueError('invalid mode')

        images_file = os.path.join(self.data_dir, mode + '.txt')

        # fn is something like `/00000_image.png`.
        self.ids = [fn.split()[0][1:6] for fn in open(images_file)]
        files_prefixes = [os.path.join(self.data_dir, id_) for id_ in self.ids]
        self.images = [id_ + '_image.png' for id_ in files_prefixes]
        self.joints = [id_ + '_joints.npy' for id_ in files_prefixes]
        self.render_light_images =\
            [id_ + '_render_light.png' for id_ in files_prefixes]
        self.bodies = [id_ + '_body.pkl' for id_ in files_prefixes]
        self.fit_crop_info = [
            open(id_ + '_fit_crop_info.txt').read() for id_ in files_prefixes]

    def __len__(self):
        return len(self.ids)

    def get_raw_data(self, i):
        img = imread(self.images[i])
        joint = np.load(self.joints[i])
        with open(self.bodies[i], 'rb') as f:
            body = pickle.load(f)
        render_light_img = imread(self.render_light_images[i])
        return img, joint, body, render_light_img, self.fit_crop_info[i]


if __name__ == '__main__':
    dataset = UniteThePeopleDataset()
    img, joint, body, render_light_img, crop_info = dataset.get_raw_data(0)
    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(render_light_img)
    plt.show()
