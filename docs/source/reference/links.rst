Links
=====


Model
-----


Feature Extraction
~~~~~~~~~~~~~~~~~~
Feature extraction models can be used to extract feature(s) given images.

.. toctree::

   links/vgg


.. autoclass:: chainercv.links.SequentialFeatureExtractor
   :members:

.. autoclass:: chainercv.links.FeaturePredictor


Detection
~~~~~~~~~

Detection links share a common method :meth:`predict` to detect objects in images.
For more details, please read :func:`FasterRCNN.predict`.

.. toctree::

   links/faster_rcnn
   links/ssd


Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~

.. module:: chainercv.links.model.segnet

Semantic segmentation links share a common method :meth:`predict` to conduct semantic segmentation of images.
For more details, please read :func:`SegNetBasic.predict`.

.. toctree::

   links/segnet


Classifiers
~~~~~~~~~~~

.. toctree::

   links/classifier


Connection
----------

.. toctree::
    links/connection
