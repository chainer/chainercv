import chainer
import collections


class SequentialChain(chainer.Chain):

    def __init__(self, functions, feature_names=None):
        super(SequentialChain, self).__init__()

        if not isinstance(functions, collections.OrderedDict):
            if feature_names is not None:
                raise ValueError('`feature_names` needs to be `None` unless '
                                 '`functions` is OrderedDict.')
            functions = collections.OrderedDict(
                [(str(i), function) for i, function in enumerate(functions)])
        self._functions = functions

        if feature_names is None:
            feature_names = self._functions.keys()[-1]
        if (not isinstance(feature_names, str) and
                all([isinstance(name, str) for name in feature_names])):
            return_tuple = True
        else:
            return_tuple = False
            feature_names = [feature_names]
        self._return_tuple = return_tuple
        self._feature_names = list(feature_names)

        with self.init_scope():
            for name, function in self._functions.items():
                if isinstance(function, chainer.Link):
                    setattr(self, name, function)

    def __call__(self, x):
        features = {}
        h = x
        for name, function in self._functions.items():
            h = function(h)
            if name in self._feature_names:
                features[name] = h

        if self._return_tuple:
            features = tuple(
                [features[name] for name in self._feature_names])
        else:
            features = list(features.values())[0]
        return features

    def copy(self):
        ret = super(SequentialChain, self).copy()
        functions = []
        for name, function in self._functions.items():
            if name in self._children:
                function = ret[name]
            functions.append((name, function))
        ret.functions = collections.OrderedDict(functions)
        return ret
