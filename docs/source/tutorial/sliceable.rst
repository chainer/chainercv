Sliceable Dataset
=================

This tutorial will walk you through the features related to sliceable dataset.
We assume that readers have a basic understanding of Chainer dataset (e.g. understand :class:`chainer.dataset.DatasetMixin`).

In ChainerCV, we introduce `sliceable` feature to datasets.
SliceableT datasets support :method:`slice` that returns a sub view of the dataset.

This example that shows the basic usage.

.. code-block:: python
    # VOCBboxDataset supports sliceable feature
    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # keys() returns the names of data
    print(dataset.keys())  # ('img', 'bbox', 'label')
    # we can get an example by []
    img, bbox, label = dataset[0]

    # get a view of the first 100 examples
    view = dataset.slice[:100]
    print(len(view))  # 100

    # get a view of image and label
    view = dataset.slice[:, ('img', 'label')]
    # the view also supports sliceable, so that we can call keys()
    print(view.keys())  # ('img', 'label')
    # we can get an example by []
    img, label = view[0]


Motivation
----------
:method:`slice` returns a view of the dataset without conducting data loading,
where :method:`DatasetMixin.__getitem__` conducts :method:`get_example` for all required examples.
Users can write efficient code by this view.

This example counts the number of images that contain dogs.

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
:method:`slice` takes indices of examples as its first argument.

.. code-block:: python
    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # the view of first 100 examples
    view = dataset.slice[:100]

    # the view of last 100 examples
    view = dataset.slice[-100:]

    # the view of 3rd, 5th, and 7th examples
    view = dataset.slice[3:8:2]

    # the view of 3rd, 1st, and 4th examples
    view = dataset.slice[[3, 1, 4]]


Usage: slice along with the axis of data
----------------------------------------
:method:`slice` takes names or indices of data as its second argument.
:method:`keys` returns all available names.

.. code-block:: python
    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # the view of image
    # note that : of the first argument means all examples
    view = dataset.slice[:, 'img']
    print(view.keys())  # 'img'
    img = view[0]

    # the view of image and label
    view = dataset.slice[:, ('img', 'label')]
    print(view.keys())  # ('img', 'label')
    img, label = view[0]

    # the view of image (returns a tuple)
    view = dataset.slice[:, ('img',)]
    print(view.keys())  # ('img',)
    img, = view[0]

    # use an index instead of a name
    view = dataset.slice[:, 1]
    print(view.keys())  # 'bbox'
    bbox = view[0]

    # mixture of names and indices
    view = dataset.slice[:, (1, 'label')]
    print(view.keys())  # ('bbox', 'label')
    bbox, label = view[0]


Usage: slice along with both axes
---------------------------------

.. code-block:: python
    from chainercv.datasets import VOCBboxDataset
    dataset = VOCBboxDataset()

    # the view of label of the first 100 examples
    view = dataset.slice[:100, 'label']


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


    dataset = SampleBboxdataset()
    img, bbox, label = dataset[0]  # get_image(0) and get_annotation(0)

    view = dataset.slice[:, 'label']
    label = view[1]  # get_annotation(0)


If you have arrays of data, you can use :class:`~chainercv.chainer_experimental.datasets.sliceable.TupleDataset`.

.. code-block:: python
    import numpy as np

    from chainercv.chainer_experimental.datasets.sliceable import TupleDataset
    from chainercv.utils import generate_random_bbox

    n = 20
    imgs = np.random.uniform(0, 255, size=(n, 3, 224, 224)).astype(np.float32)
    bboxes = [generate_random_bbox(10, (224, 224), 10, 224) for _ in range(n)]
    label = np.random.randint(0, 9, size=(n, 10)).astype(np.int32)

    dataset = TupleDataset(('img', imgs), ('bbox', bboxes), ('label', labels))

    view = dataset.slice[:, 'label']
    label = view[1]
