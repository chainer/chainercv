import chainer


class AnnotationDataset(chainer.dataset.DatasetMixin):

    """A helper dataset class for :class:`Annotatedimagedatasetmixin`.

    This is a dataset class for :class:`Annotatedimagedatasetmixin`.
    This dataset returns annotations only.
    Annotations are retrived from :meth:`self._base.get_annotation`.

    Args:
        base (object): A base dataset. This dataset should implement
            :meth:`__len__` and :meth:`get_annotation`.
    """

    def __init__(self, base):
        self._base = base

    def __len__(self):
        return len(self._base)

    def get_example(self, i):
        return self._base.get_annotation(i)


class AnnotatedImageDatasetMixin(chainer.dataset.DatasetMixin):

    """A dataset mix-in for images with annotations.

    This mix-in provides :meth:`__getitem__` and :obj:`annotations`.
    User should implement :meth:`__len__`, :meth:`get_image` and
    :meth:`get_annotation`.

    Parameters:
        annotations (AnnotationDataset): An dataset that contains annotations.
            :meth:`annotations.__getitem__` calls :meth:`get_annotation`
            internally.
    """

    def get_example(self, i):
        img = self.get_image(i)
        anno = self.get_annotation(i)
        if not isinstance(anno, tuple):
            anno = (anno,)
        return (img,) + anno

    @property
    def annotations(self):
        return AnnotationDataset(self)

    def __len__(self):
        """Returns the number of examples."""
        raise NotImplementedError

    def get_image(self, i):
        """Returns the image of the i-th example."""
        raise NotImplementedError

    def get_annotation(self, i):
        """Returns the annotation(s) of the i-th example."""
        raise NotImplementedError
