import glob
import os

import numpy as np

from chainer.dataset import download

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.cityscapes.cityscapes_utils import cityscapes_labels
from chainercv.utils import read_image
from chainercv.utils import read_label


class CityscapesSemanticSegmentationDataset(GetterDataset):

    """Semantic segmentation dataset for `Cityscapes dataset`_.

    .. _`Cityscapes dataset`: https://www.cityscapes-dataset.com

    .. note::

        Please manually download the data because it is not allowed to
        re-distribute Cityscapes dataset.

    Args:
        data_dir (string): Path to the dataset directory. The directory should
            contain at least two directories, :obj:`leftImg8bit` and either
            :obj:`gtFine` or :obj:`gtCoarse`. If :obj:`auto` is given, it uses
            :obj:`$CHAINER_DATSET_ROOT/pfnet/chainercv/cityscapes` by default.
        label_resolution ({'fine', 'coarse'}): The resolution of the labels. It
            should be either :obj:`fine` or :obj:`coarse`.
        split ({'train', 'val'}): Select from dataset splits used in
            Cityscapes dataset.
        ignore_labels (bool): If :obj:`True`, the labels marked
            :obj:`ignoreInEval` defined in the original `cityscapesScripts`_
            will be replaced with :obj:`-1` in the :meth:`get_example` method.
            The default value is :obj:`True`.

    .. _`cityscapesScripts`: https://github.com/mcordts/cityscapesScripts

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`label`, ":math:`(H, W)`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"
    """

    def __init__(self, data_dir='auto', label_resolution=None, split='train',
                 ignore_labels=True):
        super(CityscapesSemanticSegmentationDataset, self).__init__()

        if data_dir == 'auto':
            data_dir = download.get_dataset_directory(
                'pfnet/chainercv/cityscapes')
        if label_resolution not in ['fine', 'coarse']:
            raise ValueError('\'label_resolution\' argment should be eighter '
                             '\'fine\' or \'coarse\'.')

        img_dir = os.path.join(data_dir, os.path.join('leftImg8bit', split))
        resol = 'gtFine' if label_resolution == 'fine' else 'gtCoarse'
        label_dir = os.path.join(data_dir, resol)
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            raise ValueError(
                'Cityscapes dataset does not exist at the expected location.'
                'Please download it from https://www.cityscapes-dataset.com/.'
                'Then place directory leftImg8bit at {} and {} at {}.'.format(
                    os.path.join(data_dir, 'leftImg8bit'), resol, label_dir))

        self.ignore_labels = ignore_labels

        self.label_paths = []
        self.img_paths = []
        city_dnames = []
        for dname in glob.glob(os.path.join(label_dir, '*')):
            if split in dname:
                for city_dname in glob.glob(os.path.join(dname, '*')):
                    for label_path in glob.glob(
                            os.path.join(city_dname, '*_labelIds.png')):
                        self.label_paths.append(label_path)
                        city_dnames.append(os.path.basename(city_dname))
        for city_dname, label_path in zip(city_dnames, self.label_paths):
            label_path = os.path.basename(label_path)
            img_path = label_path.replace(
                '{}_labelIds'.format(resol), 'leftImg8bit')
            img_path = os.path.join(img_dir, city_dname, img_path)
            self.img_paths.append(img_path)

        self.add_getter('img', self._get_image)
        self.add_getter('label', self._get_label)

    def __len__(self):
        return len(self.img_paths)

    def _get_image(self, i):
        return read_image(self.img_paths[i])

    def _get_label(self, i):
        label_orig = read_label(self.label_paths[i], dtype=np.int32)
        if self.ignore_labels:
            label_out = np.ones(label_orig.shape, dtype=np.int32) * -1
            for label in cityscapes_labels:
                if not label.ignoreInEval:
                    label_out[label_orig == label.id] = label.trainId
        else:
            label_out = label_orig
        return label_out
