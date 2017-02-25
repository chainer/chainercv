import os.path as osp
import tarfile

from chainer.dataset import download

from chainer_cv import utils


root = 'yuyu2172/chainer-cv/pascal_voc'
urls = {
    '2012': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/'
    'VOCtrainval_11-May-2012.tar',
    '2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/'
    'VOCtrainval_06-Nov-2007.tar'
}


def get_pascal_voc(year):
    if year not in urls:
        raise ValueError

    data_root = download.get_dataset_directory(root)
    base_path = osp.join(data_root, 'VOCdevkit/VOC{}'.format(year))
    if osp.exists(base_path):
        # skip downloading
        return base_path

    download_file_path = utils.cached_download(urls[year])

    with tarfile.TarFile(download_file_path, 'r') as t:
        t.extractall(data_root)
    return base_path


pascal_voc_labels = (
    'background',
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
