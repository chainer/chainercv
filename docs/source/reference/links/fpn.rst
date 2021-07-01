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


Instance Segmentation Links
---------------------------

MaskRCNNFPNResNet50
~~~~~~~~~~~~~~~~~~~
.. autoclass:: MaskRCNNFPNResNet50
   :members:

MaskRCNNFPNResNet101
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MaskRCNNFPNResNet101
   :members:


Utility
-------

FasterRCNN
~~~~~~~~~~
.. autoclass:: FasterRCNN
   :members:

FasterRCNNFPNResNet
~~~~~~~~~~~~~~~~~~~
.. autoclass:: FasterRCNNFPNResNet
   :members:


FPN
~~~
.. autoclass:: FPN
   :members:

BboxHead
~~~~~~~~
.. autoclass:: BboxHead
   :members:

RPN
~~~~
.. autoclass:: RPN
   :members:

MaskHead
~~~~~~~~
.. autoclass:: MaskHead
   :members:

segm_to_mask
~~~~~~~~~~~~
.. autofunction:: segm_to_mask


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

mask_head_loss_pre
~~~~~~~~~~~~~~~~~~
.. autofunction:: mask_head_loss_pre

mask_head_loss_post
~~~~~~~~~~~~~~~~~~~
.. autofunction:: mask_head_loss_post

mask_to_segm
~~~~~~~~~~~~
.. autofunction:: mask_to_segm
