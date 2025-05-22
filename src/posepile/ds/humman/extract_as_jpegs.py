import argparse
import io
import os
import zipfile
import cv2
import imageio.v2 as imageio
import simplepyutils as spu
from posepile.util import improc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip-path', type=str, required=True)

    args = parser.parse_args()
    with spu.ThrottledPool() as pool:
        with zipfile.ZipFile(args.zip_path, 'r') as zip_ref:
            for pbar, relpath in spu.zip_progressbar(zip_ref.namelist()):
                if not relpath.endswith('.png'):
                    continue

                pbar.set_description(relpath)
                content = zip_ref.read(relpath)
                relpath_jpeg = spu.replace_extension(relpath, '.jpg')
                pool.apply_async(save_png_as_jpeg, (content, relpath_jpeg))

def save_png_as_jpeg(bytes_content, jpeg_path):
    im = imageio.imread(bytes_content)
    improc.imwrite_jpeg(im, jpeg_path, 95)


if __name__ == '__main__':
    main()
