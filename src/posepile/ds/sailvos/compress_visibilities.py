import os

import numpy as np
import simplepyutils as spu
from posepile.paths import DATA_ROOT


def main():
    paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/sailvos/*/visible/*.npy')
    spu.parallel_map_with_progbar(compress_numpy_array, paths)


def compress_numpy_array(in_path):
    out_path = in_path.replace('.npy', '.npz')
    array = np.load(in_path)
    np.savez_compressed(out_path, visible_ids=array)
    os.remove(in_path)


if __name__ == '__main__':
    main()
