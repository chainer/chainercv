import chainer
import collections


class ExtractionChain(chainer.Chain):

    def __init__(self, layers, layer_names=None):
        super(ExtractionChain, self).__init__()

        if not isinstance(layers, collections.OrderedDict):
            layers = collections.OrderedDict(
                [(str(i), function) for i, function in enumerate(layers)])
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
        self._layer_names = list(layer_names)

        with self.init_scope():
            for name, function in self._layers.items():
                if isinstance(function, chainer.Link):
                    setattr(self, name, function)

    def __call__(self, x):
        features = {}
        h = x
        for name, function in self._layers.items():
            h = function(h)
            if name in self._layer_names:
                features[name] = h

        if self._return_tuple:
            features = tuple(
                [features[name] for name in self._layer_names])
        else:
            features = list(features.values())[0]
        return features

    def copy(self):
        ret = super(ExtractionChain, self).copy()
        layers = []
        for name, function in self._layers.items():
            if name in self._children:
                function = ret[name]
            layers.append((name, function))
        ret.layers = collections.OrderedDict(layers)
        return ret
