import os.path as osp

import imageio.v2 as imageio
import rlemasklib
import simplepyutils as spu
from posepile.paths import DATA_ROOT


def main():
    ds_root = f'{DATA_ROOT}/genebody'
    mask_paths = spu.sorted_recursive_glob(f'{ds_root}/**/mask*.png')

    encoded_masks = spu.parallel_map_with_progbar(compress_mask, mask_paths)
    mask_relpaths = [osp.relpath(p, ds_root) for p in mask_paths]
    spu.dump_pickle(dict(zip(mask_relpaths, encoded_masks)), f'{ds_root}/masks.pkl')


def compress_mask(mask_path):
    return rlemasklib.encode(imageio.imread(mask_path) > 127)


if __name__ == '__main__':
    main()
