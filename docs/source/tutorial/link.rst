Tips using Links
================

Fine-tuning
-----------

Models in ChainerCV support the argument :obj:`pretrained_model` to load pretrained weights.
This functionality is limited in the case when fine-tuning pretrained weights.
In that circumstance, the layers specific to the classes of the original dataset may need to be randomly initialized.
In this section, we give a procedure to cope with this problem.

Copying a subset of weights in a chain can be done in few lines of code.
Here is a block of code that does this.

.. code-block:: python

    # src is a model with pretrained weights
    # dst is a model randomly initialized
    # ignore_names is the name of parameters to skip
    # For the case of VGG16, this should be ['/fc7/W', '/fc7/b']
    ignore_names = []
    src_params = {p[0]: p[1] for p in src.namedparams()}
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        if name not in ignore_names:
            dst_named_param[1].array[:] = src_params[name].array[:]


Fine-tuning to a dataset with a different number of classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the number of classes of the target dataset is different from the source dataset during fine-tuning,
the names of the weights to skip can be found automatically with the following method.

.. code-block:: python

    def get_shape_mismatch_names(src, dst):
        # all parameters are assumed to be initialized
        mismatch_names = []
        src_params = {p[0]: p[1] for p in src.namedparams()}
        for dst_named_param in dst.namedparams():
            name = dst_named_param[0]
            dst_param = dst_named_param[1]
            src_param = src_params[name]
            if src_param.shape != dst_param.shape:
                mismatch_names.append(name)
        return mismatch_names

Finally, this is a complete example using SSD300.

.. code-block:: python

    from chainercv.links import SSD300
    import numpy as np

    src = SSD300(pretrained_model='voc0712')
    # the number of classes in VOC is different from 50
    dst = SSD300(n_fg_class=50)
    # initialized weights
    dst(np.zeros((1, 3, dst.insize, dst.insize), dtype=np.float32))

    # the method described above
    ignore_names = get_shape_mismatch_names(src, dst)
    src_params = {p[0]: p[1] for p in src.namedparams()}
    for dst_named_param in dst.namedparams():
        name = dst_named_param[0]
        if name not in ignore_names:
            dst_named_param[1].array[:] = src_params[name].array[:]

    # check that weights are transfered
    np.testing.assert_equal(dst.extractor.conv1_1.W.data,
                            src.extractor.conv1_1.W.data)
    # the names of the weights that are skipped
    print(ignore_names)
