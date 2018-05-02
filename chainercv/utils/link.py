import warnings

from chainercv.utils import download_model

try:
    import cv2  # NOQA
    _available = True
except ImportError:
    _available = False


def prepare_link_initialization(out_channels,
                                pretrained_model, models, fg_only,
                                default_out_channels=None,
                                check_for_cv2=True):
    if fg_only:
        key = 'n_fg_class'
    else:
        key = 'n_class'

    if pretrained_model in models:
        model = models[pretrained_model]
        pretrained_out_chs = model[key]
        if out_channels:
            if pretrained_out_chs and not out_channels != pretrained_out_chs:
                raise ValueError(
                    '{} should be {:d}'.format(key, model[key]))
        else:
            if not pretrained_out_chs:
                raise ValueError('{} must be specified'.format(key))
            out_channels = pretrained_out_chs

        path = download_model(model['url'])

        if not _available and check_for_cv2:
            warnings.warn(
                'cv2 is not installed on your environment. '
                'Pretrained models are trained with cv2. '
                'The performace may change with Pillow backend.',
                RuntimeWarning)
    elif pretrained_model:
        path = pretrained_model
    else:
        if not out_channels:
            if default_out_channels:
                out_channels = default_out_channels
            else:
                raise ValueError('{} must be specified'.format(key))
        path = None

    return out_channels, path
