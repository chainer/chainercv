import numpy as np
import os
from scipy.io import loadmat
import tarfile

import chainer
from chainer.dataset import download

from chainercv import utils
from chainercv.utils.dataset_utils import cache_load
from chainercv.utils import read_image_as_array


root = 'pfnet/chainercv/imagenet'


def _get_imagenet(urls):
    data_root = download.get_dataset_directory(root)
    # this is error prone
    if os.path.exists(os.path.join(data_root, 'train')):
        return data_root

    for key, url in urls.items():
        download_file_path = utils.cached_download(url)

        d = os.path.join(data_root, key)
        if not os.path.exists(d):
            os.makedirs(d)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, d, ext)

    # this is an extra step needed for train dataset
    train_dir = os.path.join(data_root, 'train')
    for tar_fn in os.listdir(train_dir):
        if tar_fn[-3:] == 'tar':
            with tarfile.TarFile(os.path.join(train_dir, tar_fn), 'r') as t:
                t.extractall(train_dir)
    return data_root


class ImagenetDataset(chainer.dataset.DatasetMixin):
    """ImageNet dataset used for `ILSVRC2012`_.

    .. _ILSVRC2012: http://www.image-net.org/challenges/LSVRC/2012/

    If you pass `\'auto\'` as an argument for `base_dir`, this directory
    tries to download from `urls`. If `urls` is `None` in that case, it will
    look up for dataset in the filesystem, but do not download anything.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainer_cv/imagenet`.
        urls (dict): Dict of urls. Keys correspond to type of dataset to
            download and values correspond to urls. Keys should be
            :obj:`(train, val, test, developers_kit)`.
        mode ({'train', 'val'}): select from dataset split used in ILSVRC2012.

    """

    def __init__(self, data_dir='auto', urls=None, mode='train',
                 use_cache=False, delete_cache=False):
        if urls is not None:
            assert set(urls.keys()) == set(
                ['train', 'test', 'val', 'developers_kit'])
        if data_dir == 'auto':
            data_dir = _get_imagenet(urls)
        self.data_dir = data_dir

        developers_kit_dir = os.path.join(self.data_dir, 'developers_kit')
        # keys of synsets are ilsvrc2012_id
        self.synsets = self._parse_meta_mat(developers_kit_dir)

        self.mode = mode
        if self.mode == 'train':
            train_fns_pkl = os.path.join(
                self.data_dir, 'train_image_files.pkl')
            self.fns_dict = cache_load(
                train_fns_pkl, self._get_train_image_files, delete_cache,
                use_cache,
                args=(os.path.join(data_dir, 'train'),))
        elif self.mode == 'val':
            self.fns_dict = self._get_val_image_files(
                os.path.join(self.data_dir, 'val'), developers_kit_dir)

        # dict of lists to a list
        self.fns = []
        self.labels = []
        for key, val in self.fns_dict.items():
            self.fns += val
            self.labels += [key] * len(val)

    def __len__(self):
        return len(self.fns)

    def _get_train_image_files(self, train_dir):
        wnid_to_ilsvrc_id = {
            val['WNID']: key for key, val in self.synsets.items()}
        image_fns = {}
        for fn in os.listdir(train_dir):
            synset = fn[:9]
            if synset in wnid_to_ilsvrc_id and fn[-4:] == 'JPEG':
                int_key = self.key_to_int[synset]
                if int_key not in image_fns:
                    image_fns[int_key] = []
                image_fns[int_key].append(os.path.join(train_dir, fn))
        return image_fns

    def _get_val_image_files(self, val_dir, developers_kit_dir):
        val_gt_fn = os.path.join(
            developers_kit_dir, 'ILSVRC2012_devkit_t12/data',
            'ILSVRC2012_validation_ground_truth.txt')

        val_fns = {}
        for i, l in enumerate(open(val_gt_fn)):
            key = int(l)  # starting from 1
            index = i + 1
            if key not in val_fns:
                val_fns[key] = []
            fn = os.path.join(
                val_dir, 'ILSVRC2012_val_{0:08}.JPEG'.format(index))
            val_fns[key].append(fn)
        return val_fns

    def _parse_meta_mat(self, developers_kit_dir):
        meta_mat_fn = os.path.join(
            developers_kit_dir, 'ILSVRC2012_devkit_t12/data',
            'meta.mat')

        synsets = {}
        mat = loadmat(meta_mat_fn)
        mat_synsets = mat['synsets']
        for mat_synset in mat_synsets:
            ilsvrc2012_id = mat_synset[0][0][0][0]
            synsets[ilsvrc2012_id] = {
                'WNID': mat_synset[0][1][0],
                'words': mat_synset[0][2][0],
                'gloss': mat_synset[0][3][0],
                'num_children': mat_synset[0][4][0][0],
                'children': mat_synset[0][5][0],
                'wordnet_height': mat_synset[0][6][0][0],
                'num_train_images': mat_synset[0][7][0][0]
            }

        return synsets

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image, class_id. The image is in CHW format.
        The returned image is BGR.

        Args:
            i (int): The index of the example.
        Returns:
            i-th example

        """
        img = read_image_as_array(self.fns[i])
        if img.ndim == 2:
            img = utils.gray2rgb(img)
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.transpose(2, 0, 1).astype(np.float32)

        label = self.labels[i]
        return img, label

    def get_by_class(self, class_id, i):
        """Returns the i-th example from the given class

        Note that the class_id starts from 1.
        The returned image is BGR.

        Args:
            class_id (int): The retrieved images will be in this class.
            i (int): The index of the example in the given class.

        Returns:
            i-th example from the given class.

        """
        img = read_image_as_array(self.fns_dict[class_id][i])
        if img.ndim == 2:
            img = utils.gray2rgb(img)
        img = img[:, :, ::-1]  # BGR to RGB

        img = img.transpose(2, 0, 1).astype(np.float32)
        return img

    def get_raw_data(self, i, rgb=True):
        """Returns the i-th example.

        This returns a color image and its label. The image is in HWC foramt.

        Args:
            i (int): The index of the example.
            rgb (bool): If false, the returned image will be in BGR.

        Returns:
            i-th example (image, label)

        """
        img = read_image_as_array(self.fns[i])
        if img.ndim == 2:
            img = utils.gray2rgb(img)
        if not rgb:
            img = img[:, :, ::-1]

        label = self.labels[i]
        return img, label


if __name__ == '__main__':
    urls = {
        'train': '',
        'val': '',
        'test': '',
        'developers_kit': ''
    }
    train_dataset = ImagenetDataset(
        urls=urls, use_cache=True, delete_cache=False, mode='train')
    val_dataset = ImagenetDataset(
        urls=urls, use_cache=True, delete_cache=False, mode='val')

    import matplotlib.pyplot as plt

    for i in range(3):
        i = i + 1
        img = val_dataset.get_by_class(i, 0)
        img = img.transpose(1, 2, 0).astype(np.uint8)
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        img = val_dataset.get_by_class(i, 1)
        img = img.transpose(1, 2, 0).astype(np.uint8)
        plt.subplot(2, 2, 2)
        plt.imshow(img)

        img = train_dataset.get_by_class(i, 0)
        img = img.transpose(1, 2, 0).astype(np.uint8)
        plt.subplot(2, 2, 3)
        plt.imshow(img)
        img = train_dataset.get_by_class(i, 1)
        img = img.transpose(1, 2, 0).astype(np.uint8)
        plt.subplot(2, 2, 4)
        plt.imshow(img)
        plt.show()
