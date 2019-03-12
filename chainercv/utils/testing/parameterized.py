import six

from chainer import testing


def parameterize(*params):
    """:func:`chainer.testing.parameterize` for `pytest-xdist`.

    :func:`chainer.testing.parameterize` cannot work with `pytest-xdist`
    when the params contain functions (lambdas), classes, and random values.
    This wrapper replaces the params with their indices
    and restore the original params in :meth:`setUp`.
    """

    def deco(cls):
        setUp_orig = cls.setUp

        def setUp(self):
            param = params[self._chainercv_parameterize_index]
            print('params: {}'.format(param))
            for k, v in six.iteritems(param):
                setattr(self, k, v)
            setUp_orig(self)

        cls.setUp = setUp

        params_indices = [
            {'_chainercv_parameterize_index': i} for i in range(len(params))]
        return testing.parameterize(*params_indices)(cls)

    return deco
