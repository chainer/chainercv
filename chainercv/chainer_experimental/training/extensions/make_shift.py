from chainer.training import Extension


def make_shift(attr, optimizer=None):
    """Decorator to make shift extensions.

    This decorator wraps a function and makes a shift extension.
    Base function should take :obj:`trainer` and return a new value of
    :obj:`attr`.

    Here is an example.

    >>> # define an extension that updates 'lr' attribute
    >>> @make_shift('lr')
    >>> def warmup(trainer):
    >>>     base_lr = 0.01
    >>>     rate = 0.1
    >>>
    >>>     iteration = trainer.updater.iteration
    >>>     if iteration < 1000:
    >>>         return base_lr * (rate + (1 - rate) * iteraion / 1000)
    >>>     else:
    >>>         return base_lr
    >>>
    >>> # use the extension
    >>> trainer.extend(warmup)

    Args:
        attr (str): Name of the attribute to shift.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def deco(func):
        def ext(trainer):
            opt = optimizer or trainer.updater.get_optimizer('main')
            setattr(opt, attr, func(trainer))
        ext.default_name = func.__name__
        ext.priority = Extension.priority
        ext.initialize = ext
        return ext

    return deco
