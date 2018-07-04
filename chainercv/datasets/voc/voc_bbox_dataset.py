import numpy as np
import os
import warnings

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCBboxDataset(GetterDataset):

    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/voc`.
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox` [#voc_bbox_1]_, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label` [#voc_bbox_1]_, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`difficult` (optional [#voc_bbox_2]_), ":math:`(R,)`", \
        :obj:`bool`, --

    .. [#voc_bbox_1] If :obj:`use_difficult = True`, \
        :obj:`bbox` and :obj:`label` contain difficult instances.
    .. [#voc_bbox_2] :obj:`difficult` is available \
        if :obj:`return_difficult = True`.
    """

    def __init__(self, data_dir='auto', split='train', year='2012',
                 use_difficult=False, return_difficult=False):
        super(VOCBboxDataset, self).__init__()

        if data_dir == 'auto' and year in ['2007', '2012']:
            data_dir = voc_utils.get_voc(year, split)

        if split not in ['train', 'trainval', 'val']:
            if not (split == 'test' and year == '2007'):
                warnings.warn(
                    'please pick split from \'train\', \'trainval\', \'val\''
                    'for 2012 dataset. For 2007 dataset, you can pick \'test\''
                    ' in addition to the above mentioned splits.'
                )
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        id_ = self.ids[i]
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def _get_annotations(self, i):
        id_ = self.ids[i]
        anno_path = os.path.join(self.data_dir, 'Annotations', id_ + '.xml')
        bbox, label, difficult = voc_utils.parse_voc_bbox_annotation(
            anno_path, voc_utils.voc_bbox_label_names)
        if not self.use_difficult:
            bbox = bbox[np.logical_not(difficult)]
            label = label[np.logical_not(difficult)]
            difficult = difficult[np.logical_not(difficult)]
        return bbox, label, difficult
