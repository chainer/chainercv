import warnings

import chainer

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


def prepare_pretrained_model(pretrained_model, models):
    """Select parameters based on the existence of pretrained model.

    Args:
        pretrained_model (string): Name of the pretrained weight,
            path to the pretrained weight or :obj:`None`.
        models (dict): Map from the name of the pretrained weight
            to :obj:`model`, which is a dictionary containing the
            configuration used by the selected weight.

    """
    if pretrained_model in models:
        model = models[pretrained_model]
        path = download_model(model['url'])
        preset_param = model.get('preset_param', None)
        if model.get(model['cv2'], False):
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
    elif pretrained_model:
        path = pretrained_model
        preset_param = None
    else:
        path = None
        preset_param = None

    return path, preset_param


def prepare_param(param, preset_param):
    """Select parameters based on the existence of pretrained model.

    Args:
        param (dict): Map from the name of the parameter to values.
        preset_param (dict or None): Default map from the name of the parameter
            to values.
    """

    if preset_param is not None:
        for key in param.keys():
            if key not in preset_param:
                continue

            if param[key] is None:
                param[key] = preset_param[key]
            else:
                if not param[key] == preset_param[key]:
                    raise ValueError(
                        '{} must be {}'.format(key, preset_param[key]))

    return param
