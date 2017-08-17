import glob
import os

import numpy as np

from chainer import dataset
from chainer.dataset import download
from chainercv.datasets.cityscapes.cityscapes_utils import cityscapes_labels
from chainercv.utils import read_image


class CityscapesSemanticSegmentationDataset(dataset.DatasetMixin):

    """Dataset class for a semantic segmentation task on `Cityscapes dataset`_.

    .. _`Cityscapes dataset`: https://www.cityscapes-dataset.com

    .. note::

        Please manually downalod the data because it is not allowed to
        re-distribute Cityscapes dataset.

    Args:
        img_dir (string): Path to the image directory. It should end with
            :obj`leftImg8bit`. If :obj:`None` is given, it uses
            :obj:`$CHAINER_DATSET_ROOT/pfnet/chainercv/cityscapes/leftImg8bit`
            as default.
        label_dir (string): Path to the directory which contains labels. It
            should end with either :obj`gtFine` or :obj:`gtCoarse`. If
            :obj:`None` is given, it uses
            :obj:`$CHAINER_DATSET_ROOT/pfnet/chainercv/cityscapes/gtFine`
            as default.
        split ({'train', 'val'}): Select from dataset splits used in
            Cityscapes dataset.
        ignore_labels (bool): If True, the labels marked :obj:`ignoreInEval`
            defined in the original
            `cityscapesScripts<https://github.com/mcordts/cityscapesScripts>_`
            will be replaced with :obj:`-1` in the :meth:`get_example` method.
            The default value is :obj:`True`

    """

    def __init__(self, img_dir=None, label_dir=None, split='train',
                 ignore_labels=True):
        data_root = download.get_dataset_directory(
            'pfnet/chainercv/cityscapes')
        base_path = os.path.join(data_root, 'cityscapes')
        if img_dir is None:
            img_dir = os.path.join(base_path, 'leftImg8bit')
        if label_dir is None:
            label_dir = os.path.join(base_path, 'gtFine')
        img_dir = os.path.join(img_dir, split)
        self.ignore_labels = ignore_labels

        self.label_fnames = list()
        self.img_fnames = list()
        resol = os.path.basename(label_dir)
        for dname in glob.glob(os.path.join(label_dir, '*')):
            if split in dname:
                for label_fname in glob.glob(
                        os.path.join(dname, '*', '*_labelIds.png')):
                    self.label_fnames.append(label_fname)
        for label_fname in self.label_fnames:
            img_fname = label_fname.replace(resol, 'leftImg8bit')
            img_fname = label_fname.replace('_labelIds', '')
            self.img_fnames.append(img_fname)

    def __len__(self):
        return len(self.img_fnames)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. The color image is in CHW
        format and the label image is in HW format.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of a color image and a label whose shapes are (3, H, W) and
            (H, W) respectively. H and W are height and width of the image.
            The dtype of the color image is :obj:`numpy.float32` and
            the dtype of the label image is :obj:`numpy.int32`.

        """
        img = read_image(self.img_fnames[i])
        label_orig = read_image(
            self.label_fnames[i], dtype=np.int32, color=False)[0]
        H, W = label_orig.shape
        if self.ignore_labels:
            label_out = np.ones((H, W), dtype=np.int32) * -1
            for label in cityscapes_labels:
                if not label.ignoreInEval:
                    label_out[np.where(label_orig == label.id)] = label.trainId
        else:
            label_out = label
        return img, label_out
