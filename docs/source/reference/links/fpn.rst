FPN (Feature Pyramid Networks)
==============================

.. module:: chainercv.links.model.fpn


Detection Links
---------------

FasterRCNNFPNResnet50
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNFPNResNet50
   :members:

FasterRCNNFPNResnet101
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNFPNResNet101
   :members:


Utility
-------

FasterRCNN
~~~~~~~~~~
.. autoclass:: FasterRCNN
   :members:

FPN
~~~
.. autoclass:: FPN
   :members:

BboxHead
~~~~~~~~
.. autoclass:: BboxHead
   :members:
   :special-members:  __call__

RPN
~~~~
.. autoclass:: RPN
   :members:
   :special-members:  __call__

Train-only Utility
------------------

bbox_head_loss_pre
~~~~~~~~~~~~~~~~~~
.. autofunction:: bbox_head_loss_pre

bbox_head_loss_post
~~~~~~~~~~~~~~~~~~~
.. autofunction:: bbox_head_loss_post

rpn_loss
~~~~~~~~
.. autofunction:: rpn_loss
