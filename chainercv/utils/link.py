import warnings

import chainer

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


def prepare_model_param(param, models):
    """Select parameters and weights of model.

    Args:
        param (dict): A dict that contains all arguments.
        models (dict): Map from the name of the pretrained weight
            to :obj:`model`, which is a dictionary containing the
            configuration used by the selected weight.

    """
    pretrained_model = param.pop('pretrained_model', None)
    if pretrained_model in models:
        model = models[pretrained_model]
        path = download_model(model['url'])
        if 'param' in model:
            param = {k: v if v is not None else model['param'][k]
                     for k, v in param.items()}

        if model.get('cv2', False):
            if not _cv2_available:
                warnings.warn(
                    'cv2 is not installed on your environment. '
                    'The pretrained model is trained with cv2. '
                    'The performace may change with Pillow backend.',
                    RuntimeWarning)
            if chainer.config.cv_resize_backend != 'cv2':
                warnings.warn(
                    'Although the pretrained model is trained using cv2 as '
                    'the backend of resize function, the current '
                    'setting does not use cv2 as the backend of resize '
                    'function. The performance may change due to using '
                    'different backends. To suppress this warning, set '
                    '`chainer.config.cv_resize_backend = "cv2".',
                    RuntimeWarning)
    else:
        path = pretrained_model

    return param, path
