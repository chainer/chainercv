import filelock
import numpy as np
import os

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
    with filelock.FileLock(os.path.join(
            download.get_dataset_directory('.lock'), 'voc.lock')):
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
