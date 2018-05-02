import six
import warnings

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


def prepare_link_initialization(models, param, pretrained_model):
    if pretrained_model in models:
        model = models[pretrained_model]
        for key, value in six.iteritems(model['param']):
            if value is None:
                if key not in param:
                    raise ValueError('{} must be specified'.format(key))
            else:
                if key in param:
                    if not param[key] == value:
                        raise ValueError('{} must be {:d}'.format(key, value))
                else:
                    param[key] = value

        path = download_model(model['url'])

        if not _available and model['cv2']:
            warnings.warn(
                'cv2 is not installed on your environment. '
                'Pretrained models are trained with cv2. '
                'The performace may change with Pillow backend.',
                RuntimeWarning)
    elif pretrained_model:
        path = pretrained_model

    return param, path
