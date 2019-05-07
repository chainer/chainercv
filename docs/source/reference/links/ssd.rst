SSD (Single Shot Multibox Detector)
===================================

.. module:: chainercv.links.model.ssd


Detection Links
---------------

SSD300
~~~~~~
.. autoclass:: SSD300
   :members:

SSD512
~~~~~~
.. autoclass:: SSD512
   :members:


Utility
-------

Multibox
~~~~~~~~
.. autoclass:: Multibox
   :members:

MultiboxCoder
~~~~~~~~~~~~~
.. autoclass:: MultiboxCoder
   :members:

Normalize
~~~~~~~~~
.. autoclass:: Normalize
   :members:

SSD
~~~
.. autoclass:: SSD
   :members:

VGG16
~~~~~
.. autoclass:: VGG16
   :members:

VGG16Extractor300
~~~~~~~~~~~~~~~~~
.. autoclass:: VGG16Extractor300
   :members:

VGG16Extractor512
~~~~~~~~~~~~~~~~~
.. autoclass:: VGG16Extractor512
   :members:


Train-only Utility
------------------

GradientScaling
~~~~~~~~~~~~~~~
.. autoclass:: GradientScaling

multibox_loss
~~~~~~~~~~~~~
.. autofunction:: multibox_loss

random_crop_with_bbox_constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: random_crop_with_bbox_constraints

random_distort
~~~~~~~~~~~~~~
.. autofunction:: random_distort

resize_with_random_interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: resize_with_random_interpolation
