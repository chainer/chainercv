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
    _available = True
except ImportError:
    _available = False


class COCOInstanceSegmentationDataset(GetterDataset):

    """Instance segmentation dataset for `MS COCO`_.

    .. _`MS COCO`: http://cocodataset.org/#home

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
        year ({'2014', '2017'}): Use a dataset released in :obj:`year`.
            Splits :obj:`minival` and :obj:`valminusminival` are only
            supported in year :obj:`2014`.
        use_crowded (bool): If true, use bounding boxes that are labeled as
            crowded in the original annotation.
        return_crowded (bool): If true, this dataset returns a boolean array
            that indicates whether bounding boxes are labeled as crowded
            or not. The default value is :obj:`False`.
        return_area (bool): If true, this dataset returns areas of masks
            around objects.

    """

    def __init__(
            self, data_dir='auto', split='train', year='2017',
            use_crowded=False, return_crowded=False,
            return_area=False
    ):
        if not _available:
            raise ValueError(
                'Please install pycocotools \n'
                'pip install -e \'git+https://github.com/cocodataset/coco.git'
                '#egg=pycocotools&subdirectory=PythonAPI\'')

        if year == '2017' and split in ['minival', 'valminusminival']:
            raise ValueError(
                'coco2017 dataset does not support given split: {}'
                .format(split))

        super(COCOInstanceSegmentationDataset, self).__init__()
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
            ['label', 'area', 'crowded'],
            self._get_annotations)
        keys = ('img', 'mask', 'label')
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

        label = np.array(
            [self.cat_ids.index(anno['category_id']) for anno in annotation],
            dtype=np.int32)
        area = np.array(
            [anno['area'] for anno in annotation], dtype=np.float32)
        crowded = np.array(
            [anno['iscrowd'] for anno in annotation], dtype=np.bool)

        if not self.use_crowded:
            not_crowded = np.logical_not(crowded)
            label = label[not_crowded]
            area = area[not_crowded]
            crowded = crowded[not_crowded]

        return label, area, crowded

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
