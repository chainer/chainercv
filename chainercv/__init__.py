import pkg_resources

from chainercv import chainer_experimental  # NOQA
from chainercv import datasets  # NOQA
from chainercv import evaluations  # NOQA
from chainercv import experimental  # NOQA
from chainercv import extensions  # NOQA
from chainercv import functions  # NOQA
from chainercv import links  # NOQA
from chainercv import transforms  # NOQA
from chainercv import utils  # NOQA
from chainercv import visualizations  # NOQA


__version__ = pkg_resources.get_distribution('chainercv').version


from chainer.configuration import global_config  # NOQA


global_config.cv_read_image_backend = 'cv2'
global_config.cv_resize_backend = None
global_config.cv_rotate_backend = 'cv2'
