import os
import os.path as osp

try:
    DATA_ROOT = os.environ['DATA_ROOT']
except KeyError:
    raise KeyError(
        'The DATA_ROOT environment variable is not set. '
        'Set it to the parent dir of the dataset directories.') from None

CACHE_DIR = os.getenv('CACHE_DIR', default=f'{DATA_ROOT}/cache')
