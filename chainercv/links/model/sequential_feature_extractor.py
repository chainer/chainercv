import collections
from itertools import islice

import chainer


class SequentialFeatureExtractor(chainer.Chain):

    """A feature extractor model with a single-stream forward pass.

    This class is a base class that can be used for an implementation of
    a feature extractor model.
    The link takes :obj:`layers` to specify the computation
    conducted in :meth:`__call__`.
    :obj:`layers` is a list or :obj:`collections.OrderedDict` of
    callable objects called layers, which are going to be called sequentially
    starting from the top to the end.
    A :obj:`chainer.Link` object in the sequence will be added as
    a child link of this object.

    :meth:`__call__` returns single or multiple features that are picked up
    through a stream of computation.
    These features can be specified by :obj:`layer_names`, which contains
    the names of the layer whose output is collected.
    When :obj:`layer_names` is a string, single value is returned.
    When :obj:`layer_names` is an iterable of strings, a tuple of values
    will be returned. These values are ordered in the same order of the
    strings in :obj:`layer_names`.

    .. seealso::
        :obj:`chainercv.links.model.vgg.VGG16`

    The implementation is optimized for speed and memory.
    A layer that is not needed to collect all features listed in
    :obj:`layer_names` will not be added as a child link.
    Also, this object only conducts the minimal amount of computation needed
    to collect these features.

    Args:
        layers (list or collections.OrderedDict of callables):
            Callable objects called in the forward pass.
        layer_names (string or iterable of strings):
            Names of layers whose outputs will be collected in
            the forward pass.

    """

    def __init__(self, layers, layer_names=None):
        super(SequentialFeatureExtractor, self).__init__()

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
        ret = super(SequentialFeatureExtractor, self).copy()
        layers = []
        for name, layer in self._layers.items():
            if name in self._children:
                layer = ret[name]
            layers.append((name, layer))
        ret.layers = collections.OrderedDict(layers)
        return ret
