Faster RCNN
===========

.. module:: chainercv.links.model.faster_rcnn


faster_rcnn
-----------

FasterRCNNBase
~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNBase
   :members:
   :special-members:  __call__


faster_rcnn_vgg
---------------

FasterRCNNVGG16
~~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNVGG16

VGG16FeatureExtractor
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: VGG16FeatureExtractor

VGG16RoIPoolingHead
~~~~~~~~~~~~~~~~~~~
.. autoclass:: VGG16RoIPoolingHead


utils
-----

RegionProposalNetwork
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RegionProposalNetwork
   :members:
   :special-members:  __call__

bbox2loc
~~~~~~~~
.. autofunction:: bbox2loc

generate_anchor_base
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: generate_anchor_base

loc2bbox
~~~~~~~~
.. autofunction:: loc2bbox

ProposalCreator
~~~~~~~~~~~~~~~
.. autoclass:: ProposalCreator
   :members:
   :special-members:  __call__
