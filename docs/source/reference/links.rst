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

bbox2loc
""""""""
.. autofunction:: chainercv.links.bbox2loc

generate_anchor_base
""""""""""""""""""""
.. autofunction:: chainercv.links.generate_anchor_base

loc2bbox
""""""""
.. autofunction:: chainercv.links.loc2bbox

ProposalCreator
"""""""""""""""
.. autoclass:: ProposalCreator
   :members:
   :special-members:  __call__
