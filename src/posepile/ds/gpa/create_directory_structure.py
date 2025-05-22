import os
import os.path as osp

import simplepyutils as spu
from posepile.paths import DATA_ROOT


def main():
    """Give the image files some nested directory structure, so we can browse it more easily."""
    create_structure(f'{DATA_ROOT}/gpa/Gaussian_img_jpg_new')
    create_structure(f'{DATA_ROOT}/gpa/img_jpg_new_resnet101deeplabv3humanmask')
    create_structure(f'{DATA_ROOT}/gpa/Sequence_ids/img_jpg_gaussian_750k')


def create_structure(root):
    paths = spu.sorted_recursive_glob(f'{root}/**/*.jpg')
    print(f'{len(paths)} images found.')

    for src in spu.progressbar(paths):
        n = osp.basename(src)
        dst = f'{root}/{n[:6]}/{n[6:8]}/{n}'
        if dst != src:
            spu.ensure_parent_dir_exists(dst)
            os.rename(src, dst)


if __name__ == '__main__':
    main()
