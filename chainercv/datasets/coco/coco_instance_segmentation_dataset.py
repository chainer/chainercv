from collections import defaultdict
import json
import numpy as np
import os

import chainer

from chainercv import utils

from chainercv.datasets.coco.coco_utils import get_coco

try:
    from pycocotools import mask as coco_mask
    _availabel = True
except ImportError:
    _availabel = False


class COCOInstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    """Instance segmentation dataset for `MS COCO2014`_.

    .. _`MS COCO2014`: http://mscoco.org/dataset/#detections-challenge2015

    When queried by an index, if :obj:`return_crowded == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, mask, label, crowded, area`, a tuple of an image, bounding
    boxes, masks, labels, crowdness indicators and areas of masks.
    The parameters :obj:`return_crowded` and :obj:`return_area` decide
    whether to return :obj:`crowded` and :obj:`area`.
    :obj:`crowded` is a boolean array
    that indicates whether bounding boxes are for crowd labeling.
    When there are more than ten objects from the same category,
    bounding boxes correspond to crowd of instances instead of individual
    instances. Please see more detail in the Fig. 12 (e) of the summary
    paper [#]_.

    There are total of 82,783 training and 40,504 validation images.
    'minval' split is a subset of validation images that constitutes
    5000 images in the validation images. The remaining validation
    images are called 'minvalminus'. Concrete list of image ids and
    annotations for these splits are found `here`_.

    .. _`here`: https://github.com/rbgirshick/py-faster-rcnn/tree/master/data

    .. [#] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, \
        Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, \
        C. Lawrence Zitnick, Piotr Dollar.
        `Microsoft COCO: Common Objects in Context \
        <https://arxiv.org/abs/1405.0312>`_. arXiv 2014.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/coco`.
        split ({'train', 'val', 'minival', 'valminusminival'}): Select
            a split of the dataset.
        use_crowded (bool): If true, use bounding boxes that are labeled as
            crowded in the original annotation.
        return_crowded (bool): If true, this dataset returns a boolean array
            that indicates whether bounding boxes are labeled as crowded
            or not. The default value is :obj:`False`.
        return_area (bool): If true, this dataset returns areas of masks
            around objects.

    """

    def __init__(self, data_dir='auto', split='train',
                 use_crowded=False, return_crowded=False,
                 return_area=False):
        if not _availabel:
            raise ValueError(
                'Please install pycocotools \n'
                'pip install -e \'git+https://github.com/pdollar/coco.git'
                '#egg=pycocotools&subdirectory=PythonAPI\'')

        self.use_crowded = use_crowded
        self.return_crowded = return_crowded
        self.return_area = return_area
        if split in ['val', 'minival', 'valminusminival']:
            img_split = 'val'
        else:
            img_split = 'train'
        if data_dir == 'auto':
            data_dir = get_coco(split, img_split)

        self.img_root = os.path.join(
            data_dir, 'images', '{}2014'.format(img_split))
        anno_path = os.path.join(
            data_dir, 'annotations', 'instances_{}2014.json'.format(split))

        self.data_dir = data_dir
        anno = json.load(open(anno_path, 'r'))

        self.img_props = dict()
        for img in anno['images']:
            self.img_props[img['id']] = img
        self.ids = list(self.img_props.keys())

        cats = anno['categories']
        self.cat_ids = [cat['id'] for cat in cats]

        self.anns = dict()
        self.imgToAnns = defaultdict(list)
        for ann in anno['annotations']:
            self.imgToAnns[ann['image_id']].append(ann)
            self.anns[ann['id']] = ann

    def _get_annotations(self, i):
        img_id = self.ids[i]
        # List[{'segmentation', 'area', 'iscrowd',
        #       'image_id', 'bbox', 'category_id', 'id'}]
        annotation = self.imgToAnns[img_id]
        H = self.img_props[img_id]['height']
        W = self.img_props[img_id]['width']
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
        mask = np.array([self._segm_to_mask(anno['segmentation'], (H, W))
                         for anno in annotation])
        if len(mask) == 0:
            mask = np.zeros((0, H, W), dtype=np.bool)
        crowded = np.array([ann['iscrowd']
                            for ann in annotation], dtype=np.bool)
        area = np.array([ann['area']
                         for ann in annotation], dtype=np.float32)

        # Remove invalid boxes
        bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep_mask = np.logical_and(bbox[:, 0] <= bbox[:, 2],
                                   bbox[:, 1] <= bbox[:, 3])
        keep_mask = np.logical_and(keep_mask, bbox_area > 0)
        bbox = bbox[keep_mask]
        mask = mask[keep_mask]
        label = label[keep_mask]
        crowded = crowded[keep_mask]
        area = area[keep_mask]
        return bbox, mask, label, crowded, area

    def _segm_to_mask(self, segm, size):
        # Copied from pycocotools.coco.COCO.annToMask
        H, W = size
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = coco_mask.frPyObjects(segm, H, W)
            rle = coco_mask.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = coco_mask.frPyObjects(segm, H, W)
        else:
            rle = segm
        mask = coco_mask.decode(rle)
        return mask.astype(np.bool)

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        img_id = self.ids[i]
        img_path = os.path.join(
            self.img_root, self.img_props[img_id]['file_name'])
        img = utils.read_image(img_path, dtype=np.float32, color=True)
        _, H, W = img.shape

        bbox, mask, label, crowded, area = self._get_annotations(i)

        if not self.use_crowded:
            not_crowded = np.logical_not(crowded)
            bbox = bbox[not_crowded]
            label = label[not_crowded]
            mask = mask[not_crowded]
            crowded = crowded[not_crowded]
            area = area[not_crowded]

        example = [img, bbox, mask, label]
        if self.return_crowded:
            example += [crowded]
        if self.return_area:
            example += [area]
        return tuple(example)


def _index_list_by_mask(l, mask):
    indices = np.where(mask)[0]
    l = [l[idx] for idx in indices]
    return l
