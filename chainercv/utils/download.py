from __future__ import print_function
import hashlib
import os
import shutil
import tarfile
import tempfile
import zipfile

import filelock
from six.moves.urllib import request

import sys
import time

from chainer.dataset.download import get_dataset_directory


def _reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write('\r...{}, {} MB, {} KB/s, {} seconds passed'.format(
        percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def cached_download(url):
    """Downloads a file and caches it.

    This is different from the original ``cached_download`` in that the
    download progress is reported.

    It downloads a file from the URL if there is no corresponding cache. After
    the download, this function stores a cache to the directory under the
    dataset root (see :func:`set_dataset_root`). If there is already a cache
    for the given URL, it just returns the path to the cache without
    downloading the same file.

    Args:
        url (str): URL to download from.

    Returns:
        str: Path to the downloaded file.

    """
    cache_root = get_dataset_directory('_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.exists(cache_root):
            raise RuntimeError('cannot create download cache directory')

    lock_path = os.path.join(cache_root, '_dl_lock')
    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_root, urlhash)

    with filelock.FileLock(lock_path):
        if os.path.exists(cache_path):
            return cache_path

    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = os.path.join(temp_root, 'dl')
        print('Downloading from {}...'.format(url))
        request.urlretrieve(url, temp_path, _reporthook)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)
    finally:
        shutil.rmtree(temp_root)

    return cache_path


def download_model(url):
    """Downloads a model file and puts it under model directory.

    It downloads a file from the URL and puts it under model directory.
    For exmaple, if :obj:`url` is `http://example.com/subdir/model.npz`,
    the pretrained weights file will be saved to
    `$CHAINER_DATASET_ROOT/pfnet/chainercv/models/model.npz`.
    If there is already a file at the destination path,
    it just returns the path without downloading the same file.

    Args:
        url (str): URL to download from.

    Returns:
        str: Path to the downloaded file.

    """
    root = get_dataset_directory(
        os.path.join('pfnet', 'chainercv', 'models'))
    basename = os.path.basename(url)
    path = os.path.join(root, basename)
    if not os.path.exists(path):
        cache_path = cached_download(url)
        os.rename(cache_path, path)
    return path


def extractall(file_path, destination, ext):
    """Extracts an archive file.

    This function extracts an archive file to a destination.

    Args:
        file_path (str): The path of a file to be extracted.
        destination (str): A directory path. The archive file
            will be extracted under this directory.
        ext (str): An extension suffix of the archive file.
            This function supports :obj:`'.zip'`, :obj:`'.tar'`,
            :obj:`'.gz'` and :obj:`'.tgz'`.

    """

    if ext == '.zip':
        with zipfile.ZipFile(file_path, 'r') as z:
            z.extractall(destination)
    elif ext == '.tar':
        with tarfile.TarFile(file_path, 'r') as t:
            t.extractall(destination)
    elif ext == '.gz' or ext == '.tgz':
        with tarfile.open(file_path, 'r:gz') as t:
            t.extractall(destination)
