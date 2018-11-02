from collections import defaultdict
import json
import numpy as np
import os

from chainercv import utils

from chainercv.datasets.coco.coco_utils import get_coco

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset


class COCOBboxDataset(GetterDataset):

    """Bounding box dataset for `MS COCO`_.

    .. _`MS COCO`: http://cocodataset.org/#home

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/coco`.
        split ({'train', 'val', 'minival', 'valminusminival'}): Select
            a split of the dataset.
        year ({'2014', '2017'}): Use a dataset released in :obj:`year`.
            Splits :obj:`minival` and :obj:`valminusminival` are only
            supported in year :obj:`2014`.
        use_crowded (bool): If true, use bounding boxes that are labeled as
            crowded in the original annotation. The default value is
            :obj:`False`.
        return_area (bool): If true, this dataset returns areas of masks
            around objects. The default value is :obj:`False`.
        return_crowded (bool): If true, this dataset returns a boolean array
            that indicates whether bounding boxes are labeled as crowded
            or not. The default value is :obj:`False`.

    This dataset returns the following data.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
        :obj:`bbox` [#coco_bbox_1]_, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
        :obj:`label` [#coco_bbox_1]_, ":math:`(R,)`", :obj:`int32`, \
        ":math:`[0, \#fg\_class - 1]`"
        :obj:`area` [#coco_bbox_1]_ [#coco_bbox_2]_, ":math:`(R,)`", \
        :obj:`float32`, --
        :obj:`crowded` [#coco_bbox_3]_, ":math:`(R,)`", :obj:`bool`, --

    .. [#coco_bbox_1] If :obj:`use_crowded = True`, :obj:`bbox`, \
        :obj:`label` and :obj:`area` contain crowded instances.
    .. [#coco_bbox_2] :obj:`area` is available \
        if :obj:`return_area = True`.
    .. [#coco_bbox_3] :obj:`crowded` is available \
        if :obj:`return_crowded = True`.

    When there are more than ten objects from the same category,
    bounding boxes correspond to crowd of instances instead of individual
    instances. Please see more detail in the Fig. 12 (e) of the summary
    paper [#]_.

    .. [#] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, \
        Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, \
        C. Lawrence Zitnick, Piotr Dollar.
        `Microsoft COCO: Common Objects in Context \
        <https://arxiv.org/abs/1405.0312>`_. arXiv 2014.

    """

    def __init__(self, data_dir='auto', split='train', year='2017',
                 use_crowded=False, return_area=False, return_crowded=False):
        if year == '2017' and split in ['minival', 'valminusminival']:
            raise ValueError(
                'coco2017 dataset does not support given split: {1}'
                .format(year, split))

        super(COCOBboxDataset, self).__init__()
        self.use_crowded = use_crowded
        if split in ['val', 'minival', 'valminusminival']:
            img_split = 'val'
        else:
            img_split = 'train'
        if data_dir == 'auto':
            data_dir = get_coco(split, img_split, year)

        self.img_root = os.path.join(
            data_dir, 'images', '{}{}'.format(img_split, year))
        anno_path = os.path.join(
            data_dir, 'annotations', 'instances_{}{}.json'.format(split, year))

        self.data_dir = data_dir
        annos = json.load(open(anno_path, 'r'))

        self.id_to_prop = {}
        for prop in annos['images']:
            self.id_to_prop[prop['id']] = prop
        self.ids = sorted(list(self.id_to_prop.keys()))

        self.cat_ids = [cat['id'] for cat in annos['categories']]

        self.id_to_anno = defaultdict(list)
        for anno in annos['annotations']:
            self.id_to_anno[anno['image_id']].append(anno)

        self.add_getter('img', self._get_image)
        self.add_getter(['bbox', 'label', 'area', 'crowded'],
                        self._get_annotations)

        keys = ('img', 'bbox', 'label')
        if return_area:
            keys += ('area',)
        if return_crowded:
            keys += ('crowded',)
        self.keys = keys

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.img_root, self.id_to_prop[self.ids[i]]['file_name'])
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        return img

    def _get_annotations(self, i):
        # List[{'segmentation', 'area', 'iscrowd',
        #       'image_id', 'bbox', 'category_id', 'id'}]
        annotation = self.id_to_anno[self.ids[i]]
        bbox = np.array([ann['bbox'] for ann in annotation],
                        dtype=np.float32)
        if len(bbox) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)
        # (x, y, width, height)  -> (x_min, y_min, x_max, y_max)
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        # (x_min, y_min, x_max, y_max) -> (y_min, x_min, y_max, x_max)
        bbox = bbox[:, [1, 0, 3, 2]]

        label = np.array([self.cat_ids.index(ann['category_id'])
                          for ann in annotation], dtype=np.int32)

        area = np.array([ann['area']
                         for ann in annotation], dtype=np.float32)

        crowded = np.array([ann['iscrowd']
                            for ann in annotation], dtype=np.bool)

        # Remove invalid boxes
        bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep_mask = np.logical_and(bbox[:, 0] <= bbox[:, 2],
                                   bbox[:, 1] <= bbox[:, 3])
        keep_mask = np.logical_and(keep_mask, bbox_area > 0)

        if not self.use_crowded:
            keep_mask = np.logical_and(keep_mask, np.logical_not(crowded))

        bbox = bbox[keep_mask]
        label = label[keep_mask]
        area = area[keep_mask]
        crowded = crowded[keep_mask]
        return bbox, label, area, crowded
