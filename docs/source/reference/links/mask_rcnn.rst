Mask R-CNN
==========

.. module:: chainercv.links.model.mask_rcnn


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

MaskRCNN
~~~~~~~~
.. autoclass:: MaskRCNN
   :members:

MaskHead
~~~~~~~~
.. autoclass:: MaskHead
   :members:
   :special-members: __call__


Train-only Utility
------------------

mask_loss_pre
~~~~~~~~~~~~~
.. autofunction:: mask_loss_pre

mask_loss_post
~~~~~~~~~~~~~~
.. autofunction:: mask_loss_post
