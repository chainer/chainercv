import os
import os.path as osp
import pickle


def cache_load(cache_pkl_fn, creator, delete_cache, use_cache, args=()):
    """Cache a file if it does not exist, or loads it otherwise.

    Args:
        cache_pkl_fn (string): filename of the pkl file that contains cache
        creator (callable): a callable that creates the cache.
        delete_cache (bool): if true, the previous cache will be deleted.
        use_cache (bool): If true, the previously stored cache will be used.
            Also, if this is true, newly created contents will be cached.
        Args (tuple): arguments for creator.
    """
    if delete_cache and os.path.exists(cache_pkl_fn):
        os.remove(cache_pkl_fn)
    if use_cache and osp.exists(cache_pkl_fn):
        with open(cache_pkl_fn, 'rb') as f:
            out = pickle.load(f)
    else:
        out = creator(*args)
        if use_cache:
            with open(cache_pkl_fn, 'wb') as f:
                pickle.dump(out, f, protocol=2)
    return out
