Naming Conventions
==================


Here are the notations used.

+ :math:`B` is the size of a batch.
+ :math:`H` is the height of an image.
+ :math:`W` is the width of an image.
+ :math:`C` is the number of channels.
+ :math:`R` is the total number of instances in an image.
+ :math:`L` is the number of classes.


Data objects
~~~~~~~~~~~~

Images
""""""

+ :obj:`imgs`:  :math:`(B, C, H, W)` or :math:`[(C, H, W)]`
+ :obj:`img`:  :math:`(C, H, W)`

.. note::

    :obj:`image` is used for a name of a function or a class (e.g., :func:`chainercv.utils.write_image`).



Bounding boxes
""""""""""""""

+ :obj:`bboxes`:  :math:`(B, R, 4)` or :math:`[(R, 4)]`
+ :obj:`bbox`:  :math:`(R, 4)`
+ :obj:`bb`:  :math:`(4,)`


Labels
""""""

.. csv-table::
    :header: name, classification, detection and instance segmentation, semantic segmentation

    :obj:`labels`, ":math:`(B,)`", ":math:`(B, R)` or :math:`[(R,)]`", ":math:`(B, H, W)`"
    :obj:`label`, ":math:`()`", ":math:`(R,)`", ":math:`(H, W)`"
    :obj:`l`, r :obj:`lb`, --, ":math:`()`", --


Scores and probabilities
""""""""""""""""""""""""

score represents an unbounded confidence value.
On the other hand, probability is bounded in :obj:`[0, 1]` and sums to 1.

.. csv-table::
    :header: name, classification, detection and instance segmentation, semantic segmentation

    :obj:`scores` or :obj:`probs`, ":math:`(B, L)`", ":math:`(B, R, L)` or :math:`[(R, L)]`", ":math:`(B, L, H, W)`"
    :obj:`score` or :obj:`prob`, ":math:`(L,)`", ":math:`(R, L)`", ":math:`(L, H, W)`"
    :obj:`sc` or :obj:`pb`, --, ":math:`(L,)`", --

.. note::

    Even for objects that satisfy the definition of probability, they can be named as :obj:`score`.



Instance segmentations
""""""""""""""""""""""

+ :obj:`masks`:  :math:`(B, R, H, W)` or :math:`[(R, H, W)]`
+ :obj:`mask`:  :math:`(R, H, W)`
+ :obj:`msk`:  :math:`(H, W)`


Attributing an additonal meaning to a basic data object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RoIs
""""

+ :obj:`rois`: :math:`(R', 4)`, which consists of bounding boxes for multiple images. \
  Assuming that there are :math:`B` images each containing :math:`R_i` bounding boxes, \
  the formula :math:`R' = \sum R_i` is true.
+ :obj:`roi_indices`: An array of shape :math:`(R',)` that contains batch indices of images to which bounding boxes correspond.
+ :obj:`roi`: :math:`(R, 4)`. This is RoIs for single image.

Attributes associated to RoIs
"""""""""""""""""""""""""""""

RoIs may have additional attributes, such as class scores and masks.
These attributes are named by appending :obj:`roi_` (e.g., :obj:`scores`-like object is named as :obj:`roi_scores`).

+ :obj:`roi_xs`: :math:`(R',) + x_{shape}`
+ :obj:`roi_x`: :math:`(R,) + x_{shape}`

In the case of :obj:`scores` with shape :math:`(L,)`, :obj:`roi_xs` would have shape :math:`(R', L)`.

.. note::
   
   :obj:`roi_nouns = roi_noun = noun` when :obj:`batchsize=1`.
   Changing names interchangeably is fine.


Class-wise vs class-independent
"""""""""""""""""""""""""""""""

:obj:`cls_nouns` is a multi-class version of :obj:`nouns`.
For instance, :obj:`cls_locs` is :math:`(B, R, L, 4)` and :obj:`locs` is :math:`(B, R, 4)`.


.. note::

    :obj:`cls_probs` and :obj:`probs` can be used interchangeably in the case
    when there is no confusion.


Arbitrary input
"""""""""""""""

:obj:`x` is a variable whose shape can be inferred from the context.
It can be used only when there is no confusion on its shape.
This is usually the case when naming an input to a neural network.
