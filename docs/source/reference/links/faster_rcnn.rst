Faster R-CNN
============

.. module:: chainercv.links.model.faster_rcnn


Detection Link
--------------

FasterRCNNVGG16
~~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNVGG16


Utility
-------

bbox2loc
~~~~~~~~
.. autofunction:: bbox2loc

FasterRCNN
~~~~~~~~~~
.. autoclass:: FasterRCNN
   :members:
   :special-members:  __call__

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

RegionProposalNetwork
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RegionProposalNetwork
   :members:
   :special-members:  __call__

VGG16FeatureExtractor
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: VGG16FeatureExtractor

VGG16RoIHead
~~~~~~~~~~~~
.. autoclass:: VGG16RoIHead


Train-only Utility
------------------

AnchorTargetCreator
~~~~~~~~~~~~~~~~~~~
.. autoclass:: AnchorTargetCreator
   :members:
   :special-members:  __call__

FasterRCNNTrainChain
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNTrainChain
   :members:
   :special-members:  __call__

ProposalTargetCreator
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ProposalTargetCreator
   :members:
   :special-members:  __call__
