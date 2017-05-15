Links
=====

.. module:: chainercv.links


Model
-----

Faster RCNN
~~~~~~~~~~~

FasterRCNNBase
""""""""""""""
.. autoclass:: FasterRCNNBase
   :members:
   :special-members:  __call__

FasterRCNNVGG16
"""""""""""""""
.. autoclass:: FasterRCNNVGG16

RegionProposalNetwork
"""""""""""""""""""""
.. autoclass:: RegionProposalNetwork
   :members:
   :special-members:  __call__

bbox_regression_target
""""""""""""""""""""""
.. autofunction:: chainercv.links.bbox_regression_target

bbox_regression_target_inv
""""""""""""""""""""""""""
.. autofunction:: chainercv.links.bbox_regression_target_inv

generate_anchor_base
""""""""""""""""""""
.. autofunction:: chainercv.links.generate_anchor_base

ProposalCreator
"""""""""""""""
.. autoclass:: ProposalCreator
   :members:
   :special-members:  __call__
