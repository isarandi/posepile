import argparse

import barecat
import posepile.util.improc as improc
import simplepyutils as spu
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--barecat-path', type=str)
    parser.add_argument('--out-path', type=str)
    spu.initialize(parser)
    are_good = []
    with spu.ThrottledPool() as pool:
        with barecat.Barecat(FLAGS.barecat_path) as reader:
            for path, data in spu.progressbar_items(reader):
                pool.apply_async(is_jpeg_good, (path, data,), callback=are_good.append)

    bad_paths = [path for path, is_good in are_good if not is_good]
    spu.write_file('\n'.join(bad_paths), FLAGS.out_path)



def is_jpeg_good(name, data):
    try:
        improc.decode_jpeg_bytes(data)
    except Exception as e:
        return name, False
    else:
        return name, True

if __name__ == '__main__':
    main()
