from collections import defaultdict
import json
import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco.coco_instances_base_dataset import \
    COCOInstancesBaseDataset
from chainercv.datasets.coco.coco_utils import get_coco
from chainercv import utils


class COCOKeypointDataset(GetterDataset):

    def __init__(self, data_dir='auto', split='train', year='2017',
                 use_crowded=False,
                 return_area=False, return_crowded=False):
        super(COCOKeypointDataset, self).__init__()
        self.use_crowded = use_crowded
        if data_dir == 'auto':
            data_dir = get_coco(split, split, year, 'instances')

        self.img_root = os.path.join(
            data_dir, 'images', '{}{}'.format(split, year))
        self.data_dir = data_dir

        point_anno_path = os.path.join(
            self.data_dir, 'annotations', 'person_keypoints_{}{}.json'.format(
                split, year))
        annos = json.load(open(point_anno_path, 'r'))

        self.id_to_prop = {}
        for prop in annos['images']:
            self.id_to_prop[prop['id']] = prop
        self.ids = sorted(list(self.id_to_prop.keys()))

        self.cat_ids = [cat['id'] for cat in annos['categories']]

        self.id_to_anno = defaultdict(list)
        for anno in annos['annotations']:
            self.id_to_anno[anno['image_id']].append(anno)

        self.add_getter('img', self._get_image)
        self.add_getter(
            ['point', 'valid', 'bbox', 'label', 'area', 'crowded'],
            self._get_annotations)
        keys = ('img', 'point', 'valid', 'bbox', 'label')
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

        point = np.array(
            [anno['keypoints'] for anno in annotation], dtype=np.float32)
        if len(point) > 0:
            x = point[:, 0::3]
            y = point[:, 1::3]
            # 0: not labeled; 1: labeled, not inside mask;
            # 2: labeled and inside mask
            v = point[:, 2::3]
            valid = v > 0
            point = np.stack((y, x), axis=2)
        else:
            point = np.empty((0, 0, 2), dtype=np.float32)
            valid = np.empty((0, 0), dtype=np.bool)

        # Remove invalid boxes
        bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep_mask = np.logical_and(bbox[:, 0] <= bbox[:, 2],
                                   bbox[:, 1] <= bbox[:, 3])
        keep_mask = np.logical_and(keep_mask, bbox_area > 0)

        if not self.use_crowded:
            keep_mask = np.logical_and(keep_mask, np.logical_not(crowded))

        point = point[keep_mask]
        valid = valid[keep_mask]
        bbox = bbox[keep_mask]
        label = label[keep_mask]
        area = area[keep_mask]
        crowded = crowded[keep_mask]
        return point, valid, bbox, label, area, crowded
