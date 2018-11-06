from chainer.training import Extension


def make_shift(attr='lr', optimizer=None):
    """Decorator to make shift extensions.

    This decorator wraps a function and makes a shift extensions.
    Base function should takes :obj:`trainer` and returns a new value of
    :obj:`attr`.

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
