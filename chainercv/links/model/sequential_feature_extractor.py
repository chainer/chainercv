import collections

import chainer


class SequentialFeatureExtractor(chainer.Chain):

    """A feature extractor model with a single-stream forward pass.

    This class is a base class that can be used for an implementation of
    a feature extractor model.
    The link takes as an argument :obj:`layers` that specifies the computation
    conducted in :meth:`__call__`.
    :obj:`layers` is a list or :obj:`collections.OrderedDict` of
    callable objects called layers, which are going to be called sequentially
    starting from the top to the end.
    A :obj:`chainer.Link` object in the sequence will be added as
    a child link of this object.

    :meth:`__call__` returns single or multiple features that are picked up
    through a stream of computation.
    These features can be specified by :obj:`layer_names`, which contains
    the names of the layers whose output is collected.
    When :obj:`layer_names` is a string, single value is returned.
    When :obj:`layer_names` is an iterable of strings, a tuple of values
    will be returned. The order of the values is the same as the order of
    the strings in :obj:`layer_names`.

    Examples:

        >>> import collections
        >>> import chainer.functions as F
        >>> import chainer.links as L
        >>> layers = collections.OrderedDict([
        >>>     ('l1', L.Linear(None, 1000)),
        >>>     ('l1_relu', F.relu),
        >>>     ('l2', L.Linear(None, 1000)),
        >>>     ('l2_relu', F.relu),
        >>>     ('l3', L.Linear(None, 10))])
        >>> model = SequentialFeatureExtractor(layers, ['l2_relu', 'l1_relu'])
        >>> # These are outputs of layer l2_relu and l1_relu.
        >>> feat1, feat2 = model(x)
        >>> # The layer_names can be dynamically changed.
        >>> model.layer_names = 'l3'
        >>> # This is an output of layer l1.
        >>> feat3 = model(x)

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

        self.layer_names = layer_names

        with self.init_scope():
            for name, layer in self._layers.items():
                if isinstance(layer, chainer.Link):
                    setattr(self, name, layer)

    @property
    def layer_names(self):
        return self._layer_names

    @layer_names.setter
    def layer_names(self, layer_names):
        if layer_names is None:
            layer_names = list(self._layers.keys())[-1]

        if (not isinstance(layer_names, str) and
                all([isinstance(name, str) for name in layer_names])):
            return_tuple = True
        else:
            return_tuple = False
            layer_names = [layer_names]
        if any([name not in self._layers for name in layer_names]):
            raise ValueError('Invalid layer name')

        self._return_tuple = return_tuple
        self._layer_names = layer_names

    def __call__(self, x):
        """Forward sequential feature extraction model.

        Args:
            x (chainer.Variable or array): Input to the network.

        Returns:
            chainer.Variable or tuple of chainer.Variable:
            The returned values are determined by :obj:`layer_names`.

        """
        # The biggest index among indices of the layers that are included
        # in self._layer_names.
        last_index = max([list(self._layers.keys()).index(name) for
                          name in self._layer_names])

        features = {}
        h = x
        for name, layer in list(self._layers.items())[:last_index + 1]:
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
