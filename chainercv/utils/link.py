import warnings

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


def prepare_pretrained_model(param, pretrained_model, models, default={}):
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
                        '{} must be {:d}'.format(key, model_param[key]))

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

    for key in param.keys():
        if param[key] is None:
            if key in default:
                param[key] = default[key]
            else:
                raise ValueError('{} must be specified'.format(key))

    return param, path
