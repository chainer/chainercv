from collections import defaultdict
import json
import numpy as np
import os
import PIL.Image
import PIL.ImageDraw

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.coco.coco_utils import get_coco
from chainercv import utils

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass


class COCOInstancesBaseDataset(GetterDataset):

    def __init__(self, data_dir='auto', split='train', year='2017',
                 use_crowded=False):
        if year == '2017' and split in ['minival', 'valminusminival']:
            raise ValueError(
                'coco2017 dataset does not support given split: {}'
                .format(split))

        super(COCOInstancesBaseDataset, self).__init__()
        self.use_crowded = use_crowded

        if split in ['val', 'minival', 'valminusminival']:
            img_split = 'val'
        else:
            img_split = 'train'
        if data_dir == 'auto':
            data_dir = get_coco(split, img_split, year, 'instances')

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
        self.add_getter('mask', self._get_mask)
        self.add_getter(
            ['bbox', 'label', 'area', 'crowded'],
            self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_path = os.path.join(
            self.img_root, self.id_to_prop[self.ids[i]]['file_name'])
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        return img

    def _get_mask(self, i):
        # List[{'segmentation', 'area', 'iscrowd',
        #       'image_id', 'bbox', 'category_id', 'id'}]
        annotation = self.id_to_anno[self.ids[i]]
        H = self.id_to_prop[self.ids[i]]['height']
        W = self.id_to_prop[self.ids[i]]['width']

        mask = []
        crowded = []
        for anno in annotation:
            msk = self._segm_to_mask(anno['segmentation'], (H, W))
            # FIXME: some of minival annotations are malformed.
            if msk.shape != (H, W):
                continue
            mask.append(msk)
            crowded.append(anno['iscrowd'])
        mask = np.array(mask, dtype=np.bool)
        crowded = np.array(crowded, dtype=np.bool)
        if len(mask) == 0:
            mask = np.zeros((0, H, W), dtype=np.bool)

        if not self.use_crowded:
            not_crowded = np.logical_not(crowded)
            mask = mask[not_crowded]
        return mask

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

    def _segm_to_mask(self, segm, size):
        # Copied from pycocotools.coco.COCO.annToMask
        H, W = size
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            mask = np.zeros((H, W), dtype=np.uint8)
            mask = PIL.Image.fromarray(mask)
            for sgm in segm:
                xy = np.array(sgm).reshape((-1, 2))
                xy = [tuple(xy_i) for xy_i in xy]
                PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
            mask = np.asarray(mask)
        elif isinstance(segm['counts'], list):
            rle = coco_mask.frPyObjects(segm, H, W)
            mask = coco_mask.decode(rle)
        else:
            mask = coco_mask.decode(segm)
        return mask.astype(np.bool)
