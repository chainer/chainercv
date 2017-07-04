Links
=====

.. module:: chainercv.links.model.faster_rcnn


Feature Extraction
------------------

Feature extraction links share a common method :meth:`predict` to extract features from images.
For more details, please read :func:`VGG16Layers.predict`.

.. toctree::

   links/vgg


.. autoclass:: chainercv.links.SequentialFeatureExtractor
   :members:


Detection
---------

Detection links share a common method :meth:`predict` to detect objects in images.
For more details, please read :func:`FasterRCNN.predict`.

.. toctree::

   links/faster_rcnn
   links/ssd


Semantic Segmentation
---------------------

.. module:: chainercv.links.model.segnet

Semantic segmentation links share a common method :meth:`predict` to conduct semantic segmentation of images.
For more details, please read :func:`SegNetBasic.predict`.

.. toctree::

   links/segnet


Classifiers
-----------

.. toctree::

   links/classifier
