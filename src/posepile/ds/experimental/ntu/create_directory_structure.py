import glob
import os
import os.path as osp

import simplepyutils as spu

from posepile.ds.experimental.ntu.main import NTU_ROOT


def main():
    """Give the image files some nested directory structure, so we can browse it more easily."""

    video_root = f'{NTU_ROOT}/nturgb+d_rgb'
    paths = glob.glob(f'{video_root}/*.avi')
    for src in spu.progressbar(paths):
        n = osp.basename(src)
        dst = f'{video_root}/{n[:4]}/{n[4:12]}/{n}'
        if dst != src:
            spu.ensure_parent_dir_exists(dst)
            os.rename(src, dst)


if __name__ == '__main__':
    main()
