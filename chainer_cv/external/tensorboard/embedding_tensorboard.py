import numpy as np
import os.path as osp
import warnings

from skimage.io import imsave


def embedding_tensorboard(features, images=None, labels=None,
                          log_dir='/tmp/chainer_cv/',
                          embedding_variable_name='embedding'):
    """Save files for embedding visualization on Tensorboard.

    After calling this function, you can visualize embedding by opening up
    Tensorboard with the following command.

    `tensorboard --logdir=log_dir`

    Note that you need to install tensorflow to use this function.

    Args:
        features (numpy.ndarray): An array of shape `(N, D)`. `N` is the
            number of features to visualize and `D` is the dimension of the
            feature vector.
        images (numpy.ndarray): An array of shape `(N, H, W, 3)` or
            `(N, H, W)`. This is thumbnail images. `N` is equal to
            the number of features.
        labels (dict of list): The key is the name of attribute. The value is
            a list whose length equals the number of features.
            (ex. `{'id': [1,2], 'label': ['cat', 'dog']})

    .. seealso::
        https://www.tensorflow.org/get_started/embedding_viz

    """
    try:
        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector

    except ImportError:
        warnings.warn('Tensorflow is not installed on your environment, '
                      'so a function embedding_tensorboard can not be used.'
                      'Please install tensorflow.\n\n'
                      '  $ pip install tensorflow\n')
        return

    sess = tf.Session()

    # The embedding variable, which needs to be stored
    embedding_var = tf.Variable(features, name=embedding_variable_name)
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(log_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    if labels is not None:
        embedding.metadata_path = osp.join(log_dir, 'metadata.tsv')
        _create_metadata(labels, embedding.metadata_path)

    if images is not None:
        assert images.shape[0] == features.shape[0]
        embedding.sprite.image_path = osp.join(log_dir, 'sprite.png')
        H, W = images.shape[1], images.shape[2]
        embedding.sprite.single_image_dim.extend([H, W])
        sprite = _images_to_sprite(images)
        imsave(embedding.sprite.image_path, sprite)

    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([embedding_var])
    saver.save(sess, osp.join(log_dir, 'embedding.ckpt'), 1)


def _images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
        data: (N, H, W, 3) image

    Returns:
        data: Properly shaped H'xW'x3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def _create_metadata(labels, metadata_file):
    """Record metadata to metadata_file

    Args:
        labels: each keyword contains list of values.
            (ex. `{'a': [1,2,3], 'b': [10, 11, 12]}`)
    """
    keys = labels.keys()
    values = zip(*labels.values())

    with open(metadata_file, 'w') as f:
        for key in keys:
            f.write('{}\t'.format(key))
        f.write('\n')
        for value in values:
            for elem in value:
                f.write('{}\t'.format(elem))
            f.write('\n')


if __name__ == '__main__':
    emb = np.random.uniform(size=(1000, 4096))
    N = emb.shape[0]
    H = 32
    W = 32
    img = np.random.uniform(low=0, high=255, size=(N, H, W, 3))
    labels = {'label': range(N)}

    embedding_tensorboard(emb, img, labels)
