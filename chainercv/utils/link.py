import warnings

import chainer

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _cv2_available = True
except ImportError:
    _cv2_available = False


def prepare_pretrained_model(param, pretrained_model, models, default={}):
    """Select parameters based on the existence of pretrained model.

    Args:
        param (dict): Map from the name of the parameter to values.
        pretrained_model (string): Name of the pretrained weight,
            path to the pretrained weight or :obj:`None`.
        models (dict): Map from the name of the pretrained weight
            to :obj:`model`, which is a dictionary containing the
            configuration used by the selected weight.

            :obj:`model` has four keys: :obj:`param`, :obj:`overwritable`,
            :obj:`url` and :obj:`cv2`.

            * **param** (*dict*): Parameters assigned to the pretrained \
                weight.
            * **overwritable** (*set*): Names of parameters that are \
                overwritable (i.e., :obj:`param[key] != model['param'][key]` \
                is accepted).
            * **url** (*string*): Location of the pretrained weight.
            * **cv2** (*bool*): If :obj:`True`, a warning is raised \
                if :obj:`cv2` is not installed.

    """
    if pretrained_model in models:
        model = models[pretrained_model]
        model_param = model.get('param', {})
        overwritable = model.get('overwritable', set())

        for key in param.keys():
            if key not in model_param:
                continue

            if param[key] is None:
                param[key] = model_param[key]
            else:
                if key not in overwritable \
                   and not param[key] == model_param[key]:
                    raise ValueError(
                        '{} must be {}'.format(key, model_param[key]))

        path = download_model(model['url'])

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
    elif pretrained_model:
        path = pretrained_model
    else:
        path = None

    for key in param.keys():
        if param[key] is None:
            if key in default:
                param[key] = default[key]
            else:
                raise ValueError('{} must be specified'.format(key))

    return param, path
