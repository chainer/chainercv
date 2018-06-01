Links
=====


Model
-----

General Chain
~~~~~~~~~~~~~

.. toctree::

    links/general_chain


Feature Extraction
~~~~~~~~~~~~~~~~~~
Feature extraction links extract feature(s) from given images.

.. toctree::

   links/resnet
   links/vgg


Detection
~~~~~~~~~

Detection links share a common method :meth:`predict` to detect objects in images.
For more details, please read :func:`FasterRCNN.predict`.

.. toctree::

   links/faster_rcnn
   links/ssd
   links/yolo


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
