import filelock
import numpy as np
import os
import xml.etree.ElementTree as ET

from chainer.dataset import download

from chainercv import utils


root = 'pfnet/chainercv/voc'
urls = {
    '2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'
    'VOCtrainval_11-May-2012.tar',
    '2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
    'VOCtrainval_06-Nov-2007.tar',
    '2007_test': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
    'VOCtest_06-Nov-2007.tar'
}


def get_voc(year, split):
    if year not in urls:
        raise ValueError
    key = year

    if split == 'test' and year == '2007':
        key = '2007_test'

    # To support ChainerMN, the target directory should be locked.
    with filelock.FileLock(os.path.join(download.get_dataset_directory(
            'pfnet/chainercv/.lock'), 'voc.lock')):
        data_root = download.get_dataset_directory(root)
        base_path = os.path.join(data_root, 'VOCdevkit/VOC{}'.format(year))
        split_file = os.path.join(
            base_path, 'ImageSets/Main/{}.txt'.format(split))
        if os.path.exists(split_file):
            # skip downloading
            return base_path

        download_file_path = utils.cached_download(urls[key])
        ext = os.path.splitext(urls[key])[1]
        utils.extractall(download_file_path, data_root, ext)
    return base_path


def parse_voc_bbox_annotation(anno_path, label_names,
                              skip_names_not_in_label_names=False):
    anno = ET.parse(anno_path)
    bbox = []
    label = []
    difficult = []
    obj = anno.find('size')
    H = int(obj.find('height').text)
    W = int(obj.find('width').text)
    for obj in anno.findall('object'):
        name = obj.find('name').text.lower().strip()
        if skip_names_not_in_label_names and name not in label_names:
            continue
        label.append(label_names.index(name))
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        bbox.append([
            int(bndbox_anno.find(tag).text) - 1
            for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        if obj.find('difficult') is not None:
            difficult.append(int(obj.find('difficult').text))

    if len(bbox) > 0:
        bbox = np.stack(bbox).astype(np.float32)
        bbox[:, 0:2] = bbox[:, 0:2].clip(0)
        bbox[:, 2] = bbox[:, 2].clip(0, H)
        bbox[:, 3] = bbox[:, 3].clip(0, W)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool)
    else:
        bbox = np.zeros((0, 4), dtype=np.float32)
        label = np.zeros((0,), dtype=np.int32)
        difficult = np.zeros((0,), dtype=np.bool)
    return bbox, label, difficult


def image_wise_to_instance_wise(label_img, inst_img):
    mask = []
    label = []
    inst_ids = np.unique(inst_img)
    for inst_id in inst_ids[inst_ids != -1]:
        msk = inst_img == inst_id
        lbl = np.unique(label_img[msk])[0] - 1

        assert inst_id != -1
        assert lbl != -1

        mask.append(msk)
        label.append(lbl)
    mask = np.array(mask).astype(np.bool)
    label = np.array(label).astype(np.int32)
    return mask, label


voc_bbox_label_names = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

voc_semantic_segmentation_label_names = (('background',) +
                                         voc_bbox_label_names)

voc_instance_segmentation_label_names = voc_bbox_label_names

# these colors are used in the original MATLAB tools
voc_semantic_segmentation_label_colors = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
)
voc_semantic_segmentation_ignore_label_color = (224, 224, 192)
