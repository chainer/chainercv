Utils
=====

.. module:: chainercv.utils


Bounding Box Utilities
----------------------

bbox_iou
~~~~~~~~
.. autofunction:: bbox_iou

non_maximum_suppression
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: non_maximum_suppression


Download Utilities
------------------

cached_download
~~~~~~~~~~~~~~~
.. autofunction:: cached_download

download_model
~~~~~~~~~~~~~~
.. autofunction:: download_model

extractall
~~~~~~~~~~
.. autofunction:: extractall


Image Utilities
---------------

read_image
~~~~~~~~~~
.. autofunction:: read_image

read_label
~~~~~~~~~~
.. autofunction:: read_label

tile_images
~~~~~~~~~~~
.. autofunction:: tile_images

write_image
~~~~~~~~~~~
.. autofunction:: write_image


Iterator Utilities
------------------

apply_to_iterator
~~~~~~~~~~~~~~~~~
.. autofunction:: apply_to_iterator

ProgressHook
~~~~~~~~~~~~
.. autoclass:: ProgressHook

unzip
~~~~~
.. autofunction:: unzip


Link Utilities 
--------------

prepare_pretrained_model
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: prepare_pretrained_model


Mask Utilities
--------------

mask_iou
~~~~~~~~
.. autofunction:: mask_iou

mask_to_bbox
~~~~~~~~~~~~
.. autofunction:: mask_to_bbox

scale_mask
~~~~~~~~~~
.. autofunction:: scale_mask


Testing Utilities
-----------------

assert_is_bbox
~~~~~~~~~~~~~~
.. autofunction:: assert_is_bbox

assert_is_bbox_dataset
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_bbox_dataset

assert_is_detection_link
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_detection_link

assert_is_image
~~~~~~~~~~~~~~~
.. autofunction:: assert_is_image

assert_is_instance_segmentation_dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_instance_segmentation_dataset

assert_is_label_dataset
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_label_dataset

assert_is_point
~~~~~~~~~~~~~~~
.. autofunction:: assert_is_point

assert_is_point_dataset
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_point_dataset

assert_is_semantic_segmentation_dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_semantic_segmentation_dataset

assert_is_semantic_segmentation_link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: assert_is_semantic_segmentation_link

ConstantStubLink
~~~~~~~~~~~~~~~~
.. autoclass:: ConstantStubLink
   :members:

generate_random_bbox
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: generate_random_bbox
