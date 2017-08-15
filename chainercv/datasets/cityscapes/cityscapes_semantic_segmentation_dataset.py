import glob
import os

import numpy as np
from chainer import dataset
from chainercv.utils import read_image

from datasets import cityscapes_labels


class CityscapesSemanticSegmentationDataset(dataset.DatasetMixin):

    """Dataset class for a semantic segmentation task on Cityscapes dataset

    .. note::

        Please download the data by yourself because Cityscapes dataset doesn't
        allow to re-distribute their data.

    Args:
        img_dir (string): Path to the image dir. It should end with
            ``leftImg8bit``.
        label_dir (string): Path to the dir which contains labels. It should
            end with either ``gtFine`` or ``gtCoarse``.
        split ({'train', 'val', 'test'}): Select from dataset splits used in
            Cityscapes dataset.
        ignore_labels (bool): If True, the labels marked ``ignoreInEval``
            defined in the original
            `cityscapesScripts<https://github.com/mcordts/cityscapesScripts>_`
            will be replaced with `-1` in the `get_example` method.

    """

    def __init__(self, img_dir, label_dir, split='train', ignore_labels=True):
        img_dir = os.path.join(img_dir, split)
        self.ignore_labels = ignore_labels

        self.label_fns, self.img_fns = [], []
        if label_dir is not None:
            resol = os.path.basename(label_dir)
            for dname in glob.glob('{}/*'.format(label_dir)):
                if split in dname:
                    for label_fn in glob.glob(
                            '{}/*/*_labelIds.png'.format(dname)):
                        self.label_fns.append(label_fn)
            for label_fn in self.label_fns:
                img_fn = label_fn.replace(resol, 'leftImg8bit')
                img_fn = img_fn.replace('_labelIds', '')
                self.img_fns.append(img_fn)
        else:
            for dname in glob.glob('{}/*'.format(img_dir)):
                if split in dname:
                    for img_fn in glob.glob(
                            '{}/*_leftImg8bit.png'.format(dname)):
                        self.img_fns.append(img_fn)

    def __len__(self):
        return len(self.img_fns)

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
        img = read_image(self.img_fns[i])
        if self.label_fns == []:
            return img
        label_orig = read_image(
            self.label_fns[i], dtype=np.int32, color=False)[0]
        H, W = label_orig.shape
        if self.ignore_labels:
            label_out = np.ones((H, W), dtype=np.int32) * -1
            for label in cityscapes_labels:
                if label.ignoreInEval:
                    label_out[np.where(label_orig == label.id)] = -1
                else:
                    label_out[np.where(label_orig == label.id)] = label.trainId
        else:
            label_out = label
        img = img.astype(np.float32)
        return img, label_out
