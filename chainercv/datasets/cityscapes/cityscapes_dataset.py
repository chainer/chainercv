import glob
import re

import chainer
import numpy as np
import PIL
from six import iterkeys

from chainercv.datasets.cityscapes import cityscapes_labels
from chainercv.transforms import resize
from chainercv.utils import read_image


class CityscapesSemanticSegmentationDataset(chainer.dataset.DatasetMixin):

    """Dataset class for a semantic segmentation task on Cityscapes `u`_.

    ..  _`u`: https://www.cityscapes-dataset.com/

    Args:
        data_dir (string): Path to the root of the dataset.
        data_type ({'gtFine', 'gtCoarse'}): Select which annotation is used.
        split ({'train', 'val', 'test'}): Select from dataset splits used in
            CityscapeSemantics.
        labels ('auto' or list[cityscapes_labels.Label]): Definition of used
            labels. If this is :obj:`auto`, it uses Cityscapes' default
            definition.
        resizing_scale (float): The images and the label images are resized by
            this scale.
    """

    def __init__(self, data_dir, data_type='gtFine',
                 split='train', labels='auto',
                 resizing_scale=1.0):
        if data_type not in ['gtFine', 'gtCoarse']:
            raise ValueError('Please pick data_type from gtFine and gtCoarse')

        self.filenames = []
        labels_glob = '{}/{}/{}/*/*_labelIds.png'.format(
            data_dir, data_type, split)
        labels_pattern = re.compile(
            r'/(\w+)_(\d+)_(\d+)_{}_labelIds\.png$'.format(data_type))
        for i, label_path in enumerate(glob.glob(labels_glob)):
            m = re.search(labels_pattern, label_path)
            city, seq_idx, frame_idx = m.groups()
            image_path = '{}/leftImg8bit/{}/{}/{}_{}_{}_leftImg8bit.png' \
                .format(data_dir, split, city, city, seq_idx, frame_idx)
            self.filenames.append((label_path, image_path))

        if labels == 'auto':
            labels = cityscapes_labels.labels

        # Although trainId=255 is ignored label in the Cityscapes script,
        # it uses -1 instead for implementation easiness and consistency.
        self.id2train_id = {label.id: (label.trainId
                                       if label.trainId != 255 else -1)
                            for label in labels}

        self.resizing_scale = resizing_scale

    def __len__(self):
        return len(self.filenames)

    def get_n_class(self):
        train_ids = set(self.id2train_id.values()) - {-1}
        return len(train_ids)

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
        if i >= len(self):
            raise IndexError('index is too large')
        img_filename, label_filename = self.filenames[i]
        img = read_image(img_filename, color=True)
        label = read_image(label_filename, dtype=np.int32, color=False)

        mapped_label = np.zeros_like(label) - 1
        for original_id in iterkeys(self.id2train_id):
            train_id = self.id2train_id[original_id]
            mapped_label[label == original_id] = train_id

        if self.resizing_scale != 1.0:
            h, w = img.shape[1:]
            size = (int(w * self.resizing_scale), int(h * self.resizing_scale))
            img = resize(img, size, PIL.Image.BILINEAR)
            mapped_label = resize(mapped_label, size, PIL.Image.NEAREST)

        return img, mapped_label[0]
