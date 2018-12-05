Sliceable Dataset
=================

This tutorial will walk you through the features related to sliceable dataset.
We assume that readers have a basic understanding of Chainer dataset (e.g. understand :class:`chainer.dataset.DatasetMixin`).

In ChainerCV, we introduce `sliceable` feature to datasets.
Sliceable datasets support :meth:`slice` that returns a view of the dataset.

This example that shows the basic usage.

.. code-block:: python

    # VOCBboxDataset supports sliceable feature
    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # keys returns the names of data
    print(dataset.keys)  # ('img', 'bbox', 'label')
    # we can get an example by []
    img, bbox, label = dataset[0]

    # get a view of the first 100 examples
    view = dataset.slice[:100]
    print(len(view))  # 100

    # get a view of image and label
    view = dataset.slice[:, ('img', 'label')]
    # the view also supports sliceable, so that we can call keys
    print(view.keys)  # ('img', 'label')
    # we can get an example by []
    img, label = view[0]


Motivation
----------
:meth:`slice` returns a view of the dataset without conducting data loading,
where :meth:`DatasetMixin.__getitem__` conducts :meth:`get_example` for all required examples.
Users can write efficient code by this view.

This example counts the number of images that contain dogs.
With the sliceable feature, we can access the label information without loading images from disk..
Therefore, the first case becomes faster.

.. code-block:: python

    import time

    from chainercv.datasets import VOCBboxDataset
    from chainercv.datasets import voc_bbox_label_names

    dataset = VOCBboxDataset()
    dog_lb = voc_bbox_label_names.index('dog')

    # with slice
    t = time.time()
    count = 0
    # get a view of label
    view = dataset.slice[:, 'label']
    for i in range(len(view)):
        # we can focus on label
        label = view[i]
        if dog_lb in label:
            count += 1
    print('w/ slice: {} secs'.format(time.time() - t))
    print('{} images contain dogs'.format(count))
    print()

    # without slice
    t = time.time()
    count = 0
    for i in range(len(dataset)):
        # img and bbox are loaded but not needed
        img, bbox, label = dataset[i]
        if dog_lb in label:
            count += 1
    print('w/o slice: {} secs'.format(time.time() - t))
    print('{} images contain dogs'.format(count))
    print()


Usage: slice along with the axis of examples
--------------------------------------------
:meth:`slice` takes indices of examples as its first argument.

.. code-block:: python

    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # the view of the first 100 examples
    view = dataset.slice[:100]

    # the view of the last 100 examples
    view = dataset.slice[-100:]

    # the view of the 3rd, 5th, and 7th examples
    view = dataset.slice[3:8:2]

    # the view of the 3rd, 1st, and 4th examples
    view = dataset.slice[[3, 1, 4]]


Also, it can take a list of booleans as its first argument.
Note that the length of the list should be the same as :obj:`len(dataset)`.

.. code-block:: python

    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # make booleans
    bboxes = dataset.slice[:, 'bbox']
    booleans = [len(bbox) >= 3 for bbox in bboxes]

    # a collection of samples that contain at least three bounding boxes
    view = dataset.slice[booleans]


Usage: slice along with the axis of data
----------------------------------------
:meth:`slice` takes names or indices of data as its second argument.
:attr:`keys` returns all available names.

.. code-block:: python

    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # the view of image
    # note that : of the first argument means all examples
    view = dataset.slice[:, 'img']
    print(view.keys)  # 'img'
    img = view[0]

    # the view of image and label
    view = dataset.slice[:, ('img', 'label')]
    print(view.keys)  # ('img', 'label')
    img, label = view[0]

    # the view of image (returns a tuple)
    view = dataset.slice[:, ('img',)]
    print(view.keys)  # ('img',)
    img, = view[0]

    # use an index instead of a name
    view = dataset.slice[:, 1]
    print(view.keys)  # 'bbox'
    bbox = view[0]

    # mixture of names and indices
    view = dataset.slice[:, (1, 'label')]
    print(view.keys)  # ('bbox', 'label')
    bbox, label = view[0]

    # use booleans
    # note that the number of booleans should be the same as len(dataset.keys)
    view = dataset.slice[:, (True, True, False)]
    print(view.keys)  # ('img', 'bbox')
    img, bbox = view[0]


Usage: slice along with both axes
---------------------------------

.. code-block:: python

    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # the view of the labels of the first 100 examples
    view = dataset.slice[:100, 'label']


Concatenate and transform
-------------------------
ChainerCV provides :class:`~chainercv.chainer_experimental.datasets.sliceable.ConcatenatedDataset`
and :class:`~chainercv.chainer_experimental.datasets.sliceable.TransformDataset`.
The difference from :class:`chainer.datasets.ConcatenatedDataset` and
:class:`chainer.datasets.TransformDataset`
is that they take sliceable dataset(s) and return a sliceable dataset.

.. code-block:: python

    from chainercv.chainer_experimental.datasets.sliceable import ConcatenatedDataset
    from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
    from chainercv.datasets import VOCBboxDataset
    from chainercv.datasets import voc_bbox_label_names

    dataset_07 = VOCBboxDataset(year='2007')
    print('07:', dataset_07.keys, len(dataset_07))  # 07: ('img', 'bbox', 'label') 2501

    dataset_12 = VOCBboxDataset(year='2012')
    print('12:', dataset_12.keys, len(dataset_12))  # 12: ('img', 'bbox', 'label') 5717

    # concatenate
    dataset_0712 = ConcatenatedDataset(dataset_07, dataset_12)
    print('0712:', dataset_0712.keys, len(dataset_0712))  # 0712: ('img', 'bbox', 'label') 8218

    # transform
    def transform(in_data):
        img, bbox, label = in_data

        dog_lb = voc_bbox_label_names.index('dog')
        bbox_dog = bbox[label == dog_lb]

        return img, bbox_dog

    # we need to specify the names of data that the transform function returns
    dataset_0712_dog = TransformDataset(dataset_0712, ('img', 'bbox_dog'), transform)
    print('0712_dog:', dataset_0712_dog.keys, len(dataset_0712_dog))  # 0712_dog: ('img', 'bbox_dog') 8218


Make your own dataset
---------------------
ChainerCV provides :class:`~chainercv.chainer_experimental.datasets.sliceable.GetterDataset`
to construct a new sliceable dataset.

This example implements a sliceable bounding box dataset.

.. code-block:: python

    import numpy as np

    from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
    from chainercv.utils import generate_random_bbox

    class SampleBboxDataset(GetterDataset):
        def __init__(self):
            super(SampleBboxDataset, self).__init__()

            # register getter method for image
            self.add_getter('img', self.get_image)
            # register getter method for bbox and label
            self.add_getter(('bbox', 'label'), self.get_annotation)

        def __len__(self):
            return 20

        def get_image(self, i):
            print('get_image({})'.format(i))
            # generate dummy image
            img = np.random.uniform(0, 255, size=(3, 224, 224)).astype(np.float32)
            return img

        def get_annotation(self, i):
            print('get_annotation({})'.format(i))
            # generate dummy annotations
            bbox = generate_random_bbox(10, (224, 224), 10, 224)
            label = np.random.randint(0, 9, size=10).astype(np.int32)
            return bbox, label

    dataset = SampleBboxDataset()
    img, bbox, label = dataset[0]  # get_image(0) and get_annotation(0)

    view = dataset.slice[:, 'label']
    label = view[1]  # get_annotation(1)


If you have arrays of data, you can use :class:`~chainercv.chainer_experimental.datasets.sliceable.TupleDataset`.

.. code-block:: python

    import numpy as np

    from chainercv.chainer_experimental.datasets.sliceable import TupleDataset
    from chainercv.utils import generate_random_bbox

    n = 20
    imgs = np.random.uniform(0, 255, size=(n, 3, 224, 224)).astype(np.float32)
    bboxes = [generate_random_bbox(10, (224, 224), 10, 224) for _ in range(n)]
    labels = np.random.randint(0, 9, size=(n, 10)).astype(np.int32)

    dataset = TupleDataset(('img', imgs), ('bbox', bboxes), ('label', labels))

    print(dataset.keys)  # ('img', 'bbox', 'label')
    view = dataset.slice[:, 'label']
    label = view[1]
