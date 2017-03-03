
from __future__ import print_function
import hashlib
import os
import shutil
import tempfile

import filelock
from six.moves.urllib import request

import time
import sys


_dataset_root = os.environ.get('CHAINER_DATASET_ROOT',
                               os.path.expanduser('~/.chainer/dataset'))


def reporthook(count, block_size, total_size):
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
    cache_root = os.path.join(_dataset_root, '_dl_cache')
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
        request.urlretrieve(url, temp_path, reporthook)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cache_path)
    finally:
        shutil.rmtree(temp_root)

    return cache_path
