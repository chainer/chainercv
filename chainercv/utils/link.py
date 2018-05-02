import six
import warnings

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


def prepare_pretrained_model(models, param, pretrained_model):
    if pretrained_model in models:
        model = models[pretrained_model]

        for key, value in six.iteritems(model.get('default', {})):
            if param.get(key, None) is None:
                param[key] = value

        for key, value in six.iteritems(model['param']):
            if value is None:
                if param.get(key, None) is None:
                    raise ValueError('{} must be specified'.format(key))
            else:
                if key in param:
                    if not param[key] == value:
                        raise ValueError('{} must be {:d}'.format(key, value))
                else:
                    param[key] = value

        path = download_model(model['url'])

        if not _available and model.get('cv2', False):
            warnings.warn(
                'cv2 is not installed on your environment. '
                'Pretrained models are trained with cv2. '
                'The performace may change with Pillow backend.',
                RuntimeWarning)
    elif pretrained_model:
        path = pretrained_model
    else:
        path = None

    return param, path
