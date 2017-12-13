from chainer.dataset import download

from chainercv import utils


urls = [
    'cls_loc': 'http://image-net.org/image/ILSVRC2015/'
               'ILSVRC2015_CLS-LOC.tar.gz',
    'det': 'http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz',
    'det_test': 'http://image-net.org/image/ILSVRC2015/'
                'ILSVRC2015_DET_test.tar.gz',
    'det_test_new': 'http://image-net.org/image/ILSVRC2015/'
                    'ILSVRC2015_DET_test_new.tar.gz'
    'info': 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
]

def get_imagenet():
    download_file_path = utils.cached_download(url)
    ext = os.path.splitext(url)[1]
    utils.extractall(download_file_path, data_root, ext)
    return base_path

