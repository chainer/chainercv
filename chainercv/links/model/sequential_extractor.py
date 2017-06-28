import collections
from itertools import islice

import chainer


class SequentialExtractor(chainer.Chain):

    def __init__(self, layers, layer_names=None):
        super(SequentialExtractor, self).__init__()

        if not isinstance(layers, collections.OrderedDict):
            layers = collections.OrderedDict(
                [('{}_{}'.format(layer.__class__.__name__, i), layer)
                 for i, layer in enumerate(layers)])
        self._layers = layers

        if layer_names is None:
            layer_names = self._layers.keys()[-1]

        if (not isinstance(layer_names, str) and
                all([isinstance(name, str) for name in layer_names])):
            return_tuple = True
        else:
            return_tuple = False
            layer_names = [layer_names]
        self._return_tuple = return_tuple
        self._layer_names = layer_names

        # Delete unnecessary layers from self._layers based on layer_names.
        # Computation is equivalent to layers = layers[:last_index + 1].
        last_index = max([list(self._layers.keys()).index(name) for
                         name in self._layer_names])
        self._layers = collections.OrderedDict(
            islice(self._layers.items(), None, last_index + 1))

        with self.init_scope():
            for name, layer in self._layers.items():
                if isinstance(layer, chainer.Link):
                    setattr(self, name, layer)

    def __call__(self, x):
        features = {}
        h = x
        for name, layer in self._layers.items():
            h = layer(h)
            if name in self._layer_names:
                features[name] = h

        if self._return_tuple:
            features = tuple(
                [features[name] for name in self._layer_names])
        else:
            features = list(features.values())[0]
        return features

    def copy(self):
        ret = super(SequentialExtractor, self).copy()
        layers = []
        for name, layer in self._layers.items():
            if name in self._children:
                layer = ret[name]
            layers.append((name, layer))
        ret.layers = collections.OrderedDict(layers)
        return ret
