import os

import chainer
from chainer.dataset import download
from chainercv import utils
from chainercv.utils import read_image

from chainercv.datasets.omniglot.omniglot_label_names import \
    omniglot_label_names


root = 'pfnet/chainercv/omniglot'
url = 'https://github.com/brendenlake/omniglot/archive/master.zip'


def get_omniglot():
    root_dir = download.get_dataset_directory(root)
    download_file_path = utils.cached_download(url)
    data_dir = os.path.join(root_dir, 'omniglot-master/python')
    if not os.path.exists(data_dir):
        utils.extractall(
            download_file_path, root_dir, os.path.splitext(url)[1])

    if not os.path.exists(os.path.join(data_dir, 'images_background')):
        train_zip = os.path.join(data_dir, 'images_background.zip')
        utils.extractall(train_zip, data_dir, '.zip')
    if not os.path.exists(os.path.join(data_dir, 'images_evaluation')):
        test_zip = os.path.join(data_dir, 'images_evaluation.zip')
        utils.extractall(test_zip, data_dir, '.zip')

    return data_dir


class OmniglotDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir='auto', split='train'):
        if data_dir == 'auto':
            data_dir = get_omniglot()

        if split == 'train':
            data_dir = os.path.join(data_dir, 'images_background')
        elif split == 'test':
            data_dir = os.path.join(data_dir, 'images_evaluation')

        self.img_filenames = {}
        self.linear_img_filenames = []
        self.linear_labels = []
        for class_name in os.listdir(data_dir):
            if class_name not in self.img_filenames:
                self.img_filenames[class_name] = {}

            for char_name in os.listdir(os.path.join(data_dir, class_name)):
                if char_name not in self.img_filenames[class_name]:
                    self.img_filenames[class_name][char_name] = []
                char_dir = os.path.join(data_dir, class_name, char_name)
                img_filenames_char = sorted([
                    os.path.join(char_dir, img_name) for
                    img_name in os.listdir(char_dir)])
                self.img_filenames[class_name][char_name] += img_filenames_char
                self.linear_img_filenames += img_filenames_char
                label = omniglot_label_names.index(
                    '{}_{}'.format(class_name, char_name))
                self.linear_labels = [label] * len(img_filenames_char)

    def __len__(self):
        return len(self.linear_img_filenames)

    def get_example(self, i):
        img = read_image(self.linear_img_filenames[i], color=False)
        label = self.linear_labels[i]
        return img, label
