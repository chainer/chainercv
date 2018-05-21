import warnings

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


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
