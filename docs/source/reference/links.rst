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

License of Pretrained Models
----------------------------
Pretrained models provided by ChainerCV rely on the following resources.
When using a pretrained model, please check the lincense of resources.

.. list-table::
    :header-rows: 1

    * - model
      - resource
    * - ResNet50/101/152 (:obj:`imagenet`)
      - * `ResNet50/101/152 (trained on ImageNet) <https://github.com/KaimingHe/deep-residual-networks#models>`_
    * - VGG16 (:obj:`imagenet`)
      - * `VGG-16 (trained on ImageNet) <http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/>`_
    * - FasterRCNNVGG16 (:obj:`imagenet`)
      - * `VGG-16 (trained on ImageNet) <http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/>`_
    * - FasterRCNNVGG16 (:obj:`voc07`/:obj:`voc0712`)
      - * `VGG-16 (trained on ImageNet) <http://www.robots.ox.ac.uk/%7Evgg/research/very_deep/>`_
        * `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
    * - SSD300/SSD512 (:obj:`imagenet`)
      - * `VGG-16 (trained on ImageNet, FC reduced) <https://github.com/weiliu89/caffe/tree/ssd#preparation>`_
    * - SSD300/SSD512 (:obj:`voc0712`)
      - * `SSD300/SSD512 (trained on PASCAL VOC 2007 and 2012) <https://github.com/weiliu89/caffe/tree/ssd#models>`_
    * - YOLOv2 (:obj:`voc0712`)
      - * `Darknet19 (trained on ImageNet) <https://pjreddie.com/darknet/yolov2/#train-voc>`_
        * `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
    * - YOLOv3 (:obj:`voc0712`)
      - * `Darknet53 (trained on ImageNet) <https://pjreddie.com/darknet/yolo/#train-voc>`_
        * `PASCAL VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
    * - SegNetBasic (:obj:`camvid`)
      - * `CamVid <https://github.com/alexgkendall/SegNet-Tutorial/>`_
    * - FCISResNet101 (:obj:`sbd`)
      - * `ResNet101 (trained on ImageNet) <https://github.com/KaimingHe/deep-residual-networks#models>`_
        * `SBD <http://home.bharathh.info/pubs/codes/SBD/download.html>`_
