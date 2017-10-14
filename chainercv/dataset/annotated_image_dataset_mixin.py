import chainer


class AnnotationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, base):
        self._base = base

    def __len__(self):
        return len(self._base)

    def get_example(self, i):
        return self._base.get_annotation(i)


class AnnotatedImageDatasetMixin(chainer.dataset.DatasetMixin):

    def get_example(self, i):
        img = self.get_image(i)
        anno = self.get_annotation(i)
        if not isinstance(anno, tuple):
            anno = (anno,)
        return (img,) + anno

    @property
    def annotations(self):
        return AnnotationDataset(self)
