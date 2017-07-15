import collections

import chainer


class SequentialFeatureExtractor(chainer.Chain):

    """A feature extractor model with a single-stream forward pass.

    This class is a base class that can be used for an implementation of
    a feature extractor model.
    Callabel objects, such as :class:`chainer.Link` and
    :class:`chainer.Function`, can be registered to this link with
    :meth:`init_scope`.
    This link keeps the order of registerations and conducts the computation
    in the same order when :meth:`__call__` is called.
    A :class:`chainer.Link` object in the sequence will be added as
    a child link of this object.

    :meth:`__call__` returns single or multiple features that are picked up
    through a stream of computation.
    These features can be specified by :obj:`layer_names`, which contains
    the names of the layers whose outputs are collected.
    When :obj:`layer_names` is a string, single value is returned.
    When :obj:`layer_names` is an iterable of strings, a tuple of values
    is returned. The order of the values is the same as the order of
    the strings in :obj:`layer_names`.
    When :obj:`layer_names` is :obj:`None`, the output of the last
    layer is returned.

    Examples:

        >>> import collections
        >>> import chainer.functions as F
        >>> import chainer.links as L
        >>> model = SequentialFeatureExtractor()
        >>> with model.init_scope():
        >>>     model.l1 = L.Linear(None, 1000)
        >>>     model.l1_relu = F.relu
        >>>     model.l2 = L.Linear(None, 1000)
        >>>     model.l2_relu = F.relu
        >>>     model.l3 = L.Linear(None, 10)
        >>> model.layer_names = ['l2_relu', 'l1_relu']
        >>> # These are outputs of layer l2_relu and l1_relu.
        >>> feat1, feat2 = model(x)
        >>> # The layer_names can be dynamically changed.
        >>> model.layer_names = 'l3'
        >>> # This is an output of layer l1.
        >>> feat3 = model(x)

    Params:
        layer_names (string or iterable of strings):
            Names of layers whose outputs will be collected in
            the forward pass.

    """

    def __init__(self):
        super(SequentialFeatureExtractor, self).__init__()
        self._order = list()

    def __setattr__(self, name, value):
        super(SequentialFeatureExtractor, self).__setattr__(name, value)
        if self.within_init_scope and callable(value):
            self._order.append(name)

    def __delattr__(self, name):
        super(SequentialFeatureExtractor, self).__delattr__(name)
        try:
            self._order.remove(name)
        except ValueError:
            pass

    @property
    def layer_names(self):
        return self._layer_names

    @layer_names.setter
    def layer_names(self, layer_names):
        if layer_names is None:
            layer_names = self._order[-1]

        if (not isinstance(layer_names, str) and
                all(isinstance(name, str) for name in layer_names)):
            return_tuple = True
        else:
            return_tuple = False
            layer_names = [layer_names]
        if any(name not in self._order for name in layer_names):
            raise ValueError('Invalid layer name')

        self._return_tuple = return_tuple
        self._layer_names = layer_names

    def __call__(self, x):
        """Forward this model.

        Args:
            x (chainer.Variable or array): Input to the model.

        Returns:
            chainer.Variable or tuple of chainer.Variable:
            The returned values are determined by :obj:`layer_names`.

        """
        # The biggest index among indices of the layers that are included
        # in self._layer_names.
        last_index = max(self._order.index(name) for name in self._layer_names)

        features = {}
        h = x
        for name in self._order[:last_index + 1]:
            h = self[name](h)
            if name in self._layer_names:
                features[name] = h

        if self._return_tuple:
            features = tuple(
                [features[name] for name in self._layer_names])
        else:
            features = list(features.values())[0]
        return features
