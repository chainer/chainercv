import glob
import os

import numpy as np

import chainer
from chainercv.datasets.flic import flic_utils
from chainercv import utils

try:
    from scipy.io import loadmat
    _scipy_available = True
except (ImportError, TypeError):
    _scipy_available = False


class FLICKeypointDataset(chainer.dataset.DatasetMixin):

    """`Frames Labaled in Cinema (FLIC)`_ dataset  with annotated keypoints.

    .. _`Frames Labaled in Cinema (FLIC)`:
        https://bensapp.github.io/flic-dataset.html

    An index corresponds to each image.

    When queried by an index, this dataset returns the corresponding
    :obj:`img, keypoint`, which is a tuple of an image and keypoints
    that indicates visible keypoints in the image.
    The data type of the two elements are :obj:`float32, float32`.

    The keypoints are packed into a two dimensional array of shape
    :math:`(K, 2)`, where :math:`K` is the number of keypoints.
    Note that :math:`K=29` in FLIC dataset. Also note that not all
    keypoints are visible in an image. When a keypoint is not visible,
    the values stored for that keypoint are :obj:`~numpy.nan`. The second axis
    corresponds to the :math:`y` and :math:`x` coordinates of the
    keypoints in the image.

    The torso bounding box is a one-dimensional array of shape :math:`(4,)`.
    The elements of the bounding box corresponds to
    :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the four attributes are
    coordinates of the top left and the bottom right vertices.
    This information can optionally be retrieved from the dataset
    by setting :obj:`return_torsobox = True`.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/flic/FLIC-full`.
        split ({'train', 'test'}): Select from dataset splits used in
            the FLIC dataset.
        return_torsobox (bool): If :obj:`True`, this returns a bounding box
            around the torso. The default value is :obj:`False`.
        use_bad (bool): If :obj:`False`, the data which have :obj:`isbad = 1`
            will be ignored. The default is :obj:`False`.
        use_unchecked (bool): If :obj:`False`, the data which have
            :obj:`isunchecked = 1` will be ignored. The default is
            :obj:`False`.

    """

    def __init__(self, data_dir='auto', split='train', return_torsobox=False,
                 use_bad=False, use_unchecked=False):
        super(FLICKeypointDataset, self).__init__()
        if split not in ['train', 'test']:
            raise ValueError(
                '\'split\' argment should be eighter \'train\' or \'test\'.')

        if not _scipy_available:
            raise ImportError(
                'scipy is needed to extract labales from the .mat file.'
                'Please install scipy:\n\n'
                '\t$pip install scipy\n\n')

        if data_dir == 'auto':
            data_dir = flic_utils.get_flic()

        img_paths = {os.path.basename(fn): fn for fn in glob.glob(
            os.path.join(data_dir, 'images', '*.jpg'))}

        label_annos = [
            'poselet_hit_idx',
            'moviename',
            'coords',
            'filepath',
            'imgdims',
            'currframe',
            'torsobox',
            'istrain',
            'istest',
            'isbad',
            'isunchecked',
        ]
        annos = loadmat(os.path.join(data_dir, 'examples.mat'))

        self.img_paths = list()
        self.keypoints = list()
        self.torsoboxes = list()
        self.return_torsobox = return_torsobox

        for label in annos['examples'][0]:
            label = {label_annos[i]: val for i, val in enumerate(label)}
            if not use_bad and int(label['isbad']) == 1:
                continue
            if not use_unchecked and int(label['isunchecked']) == 1:
                continue
            if ((split == 'train' and int(label['istrain']) == 0)
                    or (split == 'test' and int(label['istest']) == 0)):
                continue

            self.img_paths.append(img_paths[label['filepath'][0]])
            self.keypoints.append(label['coords'].T[:, ::-1])
            if return_torsobox:
                self.torsoboxes.append(label['torsobox'][0, [1, 0, 3, 2]])

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and keypoints.
            The image is in CHW format and its color channel is ordered in
            RGB.
            If :obj:`return_torsobox = True`,
            a bounding box is appended to the returned value.

        """
        img = utils.read_image(self.img_paths[i])
        keypoint = np.array(self.keypoints[i], dtype=np.float32)
        if self.return_torsobox:
            return img, keypoint, self.torsoboxes[i]
        else:
            return img, keypoint
