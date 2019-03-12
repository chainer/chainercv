import six

from chainer import testing


def parameterize(*params):
    def deco(cls):
        setUp_orig = cls.setUp

        def setUp(self):
            param = self._chainercv_parameterize_params[
                self._chainercv_parameterize_index]
            print('params: {}'.format(param))
            for k, v in six.iteritems(param):
                setattr(self, k, v)
            setUp_orig(self)

        cls._chainercv_parameterize_params = params
        cls.setUp = setUp

        params_indices = [
            {'_chainercv_parameterize_index': i} for i in range(len(params))]
        return testing.parameterize(*params_indices)(cls)

    return deco
