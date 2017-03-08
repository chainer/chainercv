import numpy as np
import os

import chainer
from chainer.dataset import download

from chainercv import utils
from chainercv.utils import read_image_as_array


root = 'pfnet/chainercv/sintel'
url = 'http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip'


def _get_sintel():
    data_root = download.get_dataset_directory(root)
    if os.path.exists(os.path.join(data_root, 'training')):
        # skip downloading
        return data_root

    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)
    return data_root


class SintelDataset(chainer.dataset.DatasetMixin):

    """Dataset class for `MPI Sintel Flow Dataset`_.

    .. _`MPI Sintel Flow Dataset`: http://sintel.is.tue.mpg.de/

    This class returns tuple of (source, target, flow) as :meth:`get_example`
    is called.
    `source` is the image of the source image and `target` is the image of
    the target image which are both in CHW format. `flow` represents optical
    flow from the source to the target whose shape is :math:`(3, H, W)`.
    :math:`H` and :math:`W` are the height and the width of images.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/sintel`.
    """

    def __init__(self, data_dir='auto'):
        if data_dir == 'auto':
            data_root = _get_sintel()
            data_dir = os.path.join(data_root, 'training')
        self.data_dir = data_dir
        self.paths = self._collect_data(data_dir)
        self.keys = self.paths.keys()

    def __len__(self):
        return len(self.paths)

    def _collect_data(self, data_dir):
        paths = {}
        flow_dir = os.path.join(data_dir, 'flow')
        for root, dirs, files in os.walk(flow_dir, topdown=False):
            for file_ in files:
                if os.path.splitext(file_)[1] == '.flo':
                    dir_name = os.path.split(root)[1]
                    frame_number = int(file_[-8:-4])
                    frame_string = 'frame_{0:04d}.png'.format(frame_number)
                    next_frame_string =\
                        'frame_{0:04d}.png'.format(frame_number + 1)
                    key = '{0}_frame_{1:04d}'.format(dir_name, frame_number)
                    paths[key] = {
                        'flow': os.path.join(root, file_),
                        'src_img': os.path.join(
                            data_dir, 'clean', dir_name, frame_string),
                        'dst_img': os.path.join(
                            data_dir, 'clean', dir_name, next_frame_string)}
        return paths

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and a label image. Both of them are in CHW
        format.
        The returned image is BGR.

        Args:
            i (int): The index of the example.

        Returns:
            i-th example.

        """
        src, dst, flow = self.get_raw_data(i)
        src = src[:, :, ::-1]  # RGB to BGR
        dst = dst[:, :, ::-1]  # RGB to BGR
        src = np.transpose(src, axes=(2, 0, 1)).astype(np.float32)
        dst = np.transpose(dst, axes=(2, 0, 1)).astype(np.float32)
        flow = flow.transpose(2, 0, 1)
        return src, dst, flow

    def get_raw_data(self, i, rgb=True):
        cur_paths = self.paths[self.keys[i]]

        src = read_image_as_array(cur_paths['src_img'])
        dst = read_image_as_array(cur_paths['dst_img'])
        flow = self._read_flow_sintel(cur_paths['flow'])

        if not rgb:
            src = src[:, :, ::-1]
            dst = dst[:, :, ::-1]
        return src, dst, flow

    def _read_flow_sintel(self, path):
        """Read .flo file in Sintel.

        Returns:
            Float32 image of shape :math:`(H, W, 3)`. The last dimension
                contains (vertical_flow, horizontal_flow, valid).

        Note:
            In the original binary, flows are stored in order of
                (horizontl_flow, vertical_flow).
                Here, they are in the opposite order.
        """

        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid  .flo file')
            else:
                W = np.asscalar(np.fromfile(f, np.int32, count=1))
                H = np.asscalar(np.fromfile(f, np.int32, count=1))

                data = np.fromfile(f, np.float32, count=2 * W * H)
                data = data.reshape(H, W, 2)

        ret = np.zeros((H, W, 3), np.float32)
        # find valid pixels
        valid = np.max(data, axis=2) < 1e9
        # last dimension:  (vertical disp, horizontal disp, valid)
        ret[:, :, 0] = data[:, :, 1]
        ret[:, :, 1] = data[:, :, 0]
        ret[:, :, 2] = valid
        return ret


if __name__ == '__main__':
    dataset = SintelDataset()
