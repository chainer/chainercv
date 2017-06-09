import numpy as np
import os

import chainer
from chainer.dataset import download

from chainercv import utils

try:
    from pycocotools.coco import COCO
    _available = True

except ImportError:
    _available = False

# How you can get the labels
# >>> coco = dataset.coco
# >>> cat_dict = coco.loadCats(coco.getCatIds())
# >>> coco_detection_labels = [c['name'] for c in cat_dict]
coco_detection_label_names = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')


root = 'pfnet/chainercv/coco'
img_urls = {
    'train': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
    'val': 'http://msvocds.blob.core.windows.net/coco2014/val2014.zip'
}
anno_urls = {
    'train': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
    'instances_train-val2014.zip',
    'val': 'http://msvocds.blob.core.windows.net/annotations-1-0-3/'
    'instances_train-val2014.zip',
    'valminusminival': 'https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/'
    'instances_valminusminival2014.json.zip',
    'minval': 'https://dl.dropboxusercontent.com/s/o43o90bna78omob/'
    'instances_minival2014.json.zip'
}


def _get_coco(split, img_split):

    url = img_urls[img_split]
    data_dir = download.get_dataset_directory(root)
    img_root = os.path.join(data_dir, 'images')
    created_img_root = os.path.join(img_root, '{}2014'.format(img_split))
    annos_root = os.path.join(data_dir, 'annotations')
    anno_fn = os.path.join(annos_root, 'instances_{}2014.json'.format(split))
    if not os.path.exists(created_img_root):
        download_file_path = utils.cached_download(url)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, img_root, ext)
    if not os.path.exists(anno_fn):
        anno_url = anno_urls[split]
        download_file_path = utils.cached_download(anno_url)
        ext = os.path.splitext(anno_url)[1]
        utils.extractall(download_file_path, data_dir, ext)
    return data_dir


class COCODetectionDataset(chainer.dataset.DatasetMixin):

    """Dataset class for the detection task of `MS COCO`_.

    .. _`MS COCO`: http://mscoco.org/dataset/#detections-challenge2015

    When queried by an index, if :obj:`return_crowded == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_crowded == True`, this dataset returns corresponding
    :obj:`img, bbox, label, crowded`. :obj:`crowded` is a boolean array
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

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :obj:`(x_min, y_min, x_max, y_max)`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`chainercv.datasets.coco_detection_label_names`.

    The array :obj:`crowded` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`crowded.dtype == numpy.bool`

    .. [#] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, \
        Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, \
        C. Lawrence Zitnick, Piotr Dollar.
        `Microsoft COCO: Common Objects in Context \
        <https://arxiv.org/abs/1405.0312>`_. arXiv 2014.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/coco`.
        split ({'train', 'val', 'minval', 'minvalminus'}): Select
            a split of the dataset.
        use_crowded (bool): If true, use bounding boxes that are labeled as
            crowded in the original annotation.
        return_crowded (bool): If true, this dataset returns a boolean array
            that indicates whether bounding boxes are labeled as crowded
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir='auto', split='train',
                 use_crowded=False, return_crowded=False):
        if not _available:
            raise ValueError(
                'Please install MS COCO dataset package. '
                'Installation can be done by the following commands \n'
                ' $ git clone https://github.com/pdollar/coco \n'
                ' $ cd PythonAPI \n'
                ' $ pip install -e . \n')

        if split in ['val', 'minval', 'minvalminus']:
            img_split = 'val'
        else:
            img_split = 'train'
        if data_dir == 'auto':
            data_dir = _get_coco(split, img_split)

        self.img_root = os.path.join(
            data_dir, 'images', '{}2014'.format(img_split))
        anno_fn = os.path.join(
            data_dir, 'annotations', 'instances_{}2014.json'.format(split))
        self.data_dir = data_dir
        self.coco = COCO(anno_fn)
        self.ids = list(self.coco.imgs.keys())
        self.cat_ids = self.coco.getCatIds()

        self.use_crowded = use_crowded
        self.return_crowded = return_crowded

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        # This method is based on the code found at rbgirshick/py-faster-rcnn.
        # https://github.com/rbgirshick/py-faster-rcnn/blob/master/
        # lib/datasets/coco.py#L228
        img_id = self.ids[i]

        img_fn = os.path.join(
            self.img_root, self.coco.loadImgs(img_id)[0]['file_name'])
        img = utils.read_image(img_fn, dtype=np.float32, color=True)
        _, H, W = img.shape

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # List[{'segmentation', 'area', 'iscrowd',
        #       'image_id', 'bbox', 'category_id', 'id'}]
        target = self.coco.loadAnns(ann_ids)
        label = np.array(
            [self.cat_ids.index(ann['category_id']) for ann in target],
            dtype=np.int32)

        bbox = np.array([ann['bbox'] for ann in target], dtype=np.float32)
        if len(bbox) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)

        # (x, y, width, height)  -> (x_min, y_min, x_max, y_max)
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        # Sanitize boxes
        bbox[:, :2] = np.maximum(bbox[:, :2], 0)
        bbox[:, 2] = np.minimum(bbox[:, 2], W)
        bbox[:, 3] = np.minimum(bbox[:, 3], H)

        # Remove invalid boxes
        area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep_mask = np.logical_and(bbox[:, 0] <= bbox[:, 2],
                                   bbox[:, 1] <= bbox[:, 3])
        keep_mask = np.logical_and(keep_mask, area > 0)
        bbox = bbox[keep_mask]
        label = label[keep_mask]

        crowded = np.array([ann['iscrowd'] for ann in target], dtype=np.bool)

        if not self.use_crowded:
            bbox = bbox[np.logical_not(crowded)]
            label = label[np.logical_not(crowded)]

        if self.return_crowded:
            return img, bbox, label, crowded
        return img, bbox, label
