import json
import os

import chainer
from chainer.dataset import download

from chainercv import utils

root = 'pfnet/chainer/visual_genome'

vg_100k_url = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip'
vg_100k_2_url = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip'
image_data_url = 'http://visualgenome.org/static/data/dataset/' \
    'image_data.json.zip'
region_descriptions_url = 'http://visualgenome.org/static/data/dataset/' \
    'region_descriptions.json.zip'


def get_visual_genome():
    """Get the default path to the Visual Genome image directory.

    Returns:
        str: A path to the image directory.

    """
    def move_files(src_dir, dst_dir):
        # Move all files in the src_dir to the dst_dir and remove the src_dir
        for f in os.listdir(src_dir):
            src = os.path.join(src_dir, f)
            if os.path.isfile(src):
                dst = os.path.join(dst_dir, f)
                os.rename(src, dst)
        os.rmdir(src_dir)

    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, 'VG_100K_ALL')

    if os.path.exists(base_path):
        return base_path

    print('Caching Visual Genome image files...')
    os.mkdir(base_path)
    move_files(_get_extract_data(vg_100k_url, data_root, 'VG_100K'), base_path)
    move_files(_get_extract_data(vg_100k_2_url, data_root, 'VG_100K_2'),
               base_path)

    return base_path


def get_image_data():
    """Get the default path to the image data JSON file.

    Returns:
        str: A path to the image data JSON file.

    """
    data_root = download.get_dataset_directory(root)
    return _get_extract_data(image_data_url, data_root, 'image_data.json')


def get_region_descriptions():
    """Get the default path to the region descriptions JSON file.

    Returns:
        str: A path to the region descriptions JSON file.

    """
    data_root = download.get_dataset_directory(root)
    return _get_extract_data(region_descriptions_url, data_root,
                             'region_descriptions.json')


class VisualGenomeDatasetBase(chainer.dataset.DatasetMixin):
    """Base class for Visual Genome dataset.

    """

    def __init__(self, data_dir='auto', image_data='auto'):
        if data_dir == 'auto':
            data_dir = get_visual_genome()
        if image_data == 'auto':
            image_data = get_image_data()
        self.data_dir = data_dir

        with open(image_data, 'r') as f:
            img_ids = [img_data['image_id'] for img_data in json.load(f)]
        self.img_ids = sorted(img_ids)

    def __len__(self):
        return len(self.img_ids)

    def get_image_id(self, i):
        return self.img_ids[i]

    def get_image(self, img_id):
        img_path = os.path.join(self.data_dir, str(img_id) + '.jpg')
        return utils.read_image(img_path, color=True)


def _get_extract_data(url, data_root, member_path):
    base_path = os.path.join(data_root, member_path)

    if os.path.exists(base_path):
        return base_path

    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)

    return base_path
