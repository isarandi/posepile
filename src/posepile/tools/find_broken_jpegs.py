import multiprocessing
import os.path as osp
import sys

import simplepyutils as spu

import posepile.util.improc as improc


def main():
    image_root = sys.argv[1]
    out_path = sys.argv[2]
    image_paths = spu.sorted_recursive_glob(f'{image_root}/**/*.jpg')
    pool = multiprocessing.Pool()
    is_readable = pool.imap(improc.is_jpeg_readable, image_paths, chunksize=256)
    is_readable = list(spu.progressbar(is_readable, total=len(image_paths)))
    broken_paths = [p for p, r in zip(image_paths, is_readable) if not r]
    broken_relpaths = [osp.relpath(p, image_root) for p in broken_paths]
    spu.write_file(out_path, '\n'.join(broken_relpaths) + '\n')


if __name__ == '__main__':
    main()
