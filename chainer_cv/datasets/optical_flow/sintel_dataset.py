import numpy as np
import os
import os.path as osp
from skimage.io import imread
import zipfile

import chainer
from chainer.dataset import download

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

    The format of correspondence data returned by `get_example` is determined
    by ``mode``.
    If ``mode`` is `flow`, it returns (source, target, flow).
    `source` is the image of the source image and `target` is the image of
    the target image which are both in CHW format. `flow` represents optical
    flow from the source to the target whose shape is (3, H, W). H and W are
    the height and the width of images.

    If `mode` is `verts`, it returns
    (source, target, source_verts, target_verts). `source_verts` and
    `target_verts` are an array of shape (n_verts, 2). `n_verts` is the
    number of pixels who appear in both the source and the target. The
    second axis contains the location of the pixel.
    `source_verts[i]` and `target_verts[i]` correspond to each other for
    arbitrary `i`.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            'auto', this class will automatically download data for you
            under ``$CHAINER_DATASET_ROOT/pfnet/chainer_cv/sintel``.
        mode (string, {'flow', 'verts'}): Determines the format of
            correspondence data between the source and the destination image.

    """

    def __init__(self, data_dir='auto', mode='flow'):
        if data_dir == 'auto':
            data_root = _get_sintel()
            data_dir = osp.join(data_root, 'training')
        self.data_dir = data_dir
        self.paths = self._collect_data(data_dir)
        self.keys = self.paths.keys()

        self.mode = mode

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
        """Returns the i-th example.

        Returns a color image and a label image. Both of them are in CHW
        format.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example

        """
        src, dst, flow = self.get_raw_data(i)
        src = np.transpose(src, axes=(2, 0, 1)).astype(np.float32)
        dst = np.transpose(dst, axes=(2, 0, 1)).astype(np.float32)
        if self.mode == 'flow':
            flow = flow.transpose(2, 0, 1)
            return src, dst, flow
        elif self.mode == 'verts':
            verts = corresp.flow2verts(flow)
            src_verts = verts[0]
            dst_verts = verts[1]
            return src, dst, src_verts, dst_verts
        else:
            raise ValueError('mode is either \'flow\' or \'verts\'')

    def get_raw_data(self, i):
        cur_paths = self.paths[self.keys[i]]

        src_img = imread(cur_paths['src_img'])
        dst_img = imread(cur_paths['dst_img'])
        flow = corresp.read_flow_sintel(cur_paths['flow'])
        return src_img, dst_img, flow
