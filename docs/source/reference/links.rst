Links
=====

.. module:: chainercv.links.model.faster_rcnn


Classification
--------------

Classification links share a common method :meth:`predict` to classify or extract features with images.
For more details, please read :func:`VGG16Layers.predict`.

.. toctree::

   links/vgg

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
