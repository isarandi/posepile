import os
import os.path as osp
import os.path
import sys

import imageio.v2 as imageio
import simplepyutils as spu


def main():
    pattern = sys.argv[1]
    image_paths = spu.sorted_recursive_glob(pattern)
    spu.parallel_map_with_progbar(to_jpg, image_paths)


def to_jpg(in_path):
    im = imageio.imread(in_path)
    in_path_noext, ext = osp.splitext(in_path)
    imageio.imwrite(in_path_noext + '.jpg', im, quality=85)
    os.remove(in_path)


if __name__ == '__main__':
    main()
