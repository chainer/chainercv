import os
import os.path as osp
import zipfile

import chainer
from chainer.dataset import download
import numpy as np
from skimage.io import imread

from chainer_cv import utils
import corresp


root = 'pfnet/chainer_cv/sintel'
url = 'http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip'

def _get_sintel():
    data_root = download.get_dataset_directory(root)
    if osp.exists(osp.join(data_root, 'training')):
        # skip downloading
        return data_root

    download_file_path = utils.cached_download(url)
 
    with zipfile.ZipFile(download_file_path, 'r') as z:
        z.extractall(data_root)
    return data_root


class SintelDataset(chainer.dataset.DatasetMixin):

    """Simple class to load data from MPI Sintel Dataset

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/pfnet/chainer_cv/sintel``.
    """

    def __init__(self, data_dir='auto'):
        if data_dir == 'auto':
            data_root = _get_sintel()
            data_dir = osp.join(data_root, 'training')
        self.data_dir = data_dir
        self.paths = self._collect_data(data_dir)
        self.keys = self.paths.keys()

    def __len__(self):
        return len(self.paths)

    def _collect_data(self, data_dir):
        paths = {}
        flow_dir = osp.join(data_dir, 'flow')
        for root, dirs, files in os.walk(flow_dir, topdown=False):
            for file_ in files:
                if osp.splitext(file_)[1] == '.flo':
                    dir_name = osp.split(root)[1]
                    frame_number = int(file_[-8:-4])
                    frame_string = 'frame_{0:04d}.png'.format(frame_number)
                    next_frame_string =\
                        'frame_{0:04d}.png'.format(frame_number + 1)
                    paths['{0}_frame_{1:04d}'.format(dir_name, frame_number)] = {
                        'flow': osp.join(root, file_),
                        'src_img': osp.join(
                            data_dir, 'clean', dir_name, frame_string),
                        'dst_img': osp.join(
                            data_dir, 'clean', dir_name, next_frame_string)}
        return paths

    def get_example(self, i):
        src, dst, flow = self.get_raw_data(i)
        src = np.transpose(src, axes=(2, 0, 1)).astype(np.float32)
        dst = np.transpose(dst, axes=(2, 0, 1)).astype(np.float32)
        return src, dst, flow

    def get_raw_data(self, i):
        cur_paths = self.paths[self.keys[i]]

        src_img = imread(cur_paths['src_img'])
        dst_img = imread(cur_paths['dst_img'])
        flow = corresp.read_flow_sintel(cur_paths['flow'])
        return src_img, dst_img, flow


if __name__ == '__main__':
    sintel_dataset = SintelDataset()
    src_img, dst_img, flow = sintel_dataset.get_example(0)
