import glob
import os
import os.path as osp
import shutil

import cv2
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
from PIL import Image

from posepile.paths import DATA_ROOT


def main():
    image_paths = glob.glob(f'{DATA_ROOT}/inria_holidays/jpg/*')
    os.makedirs(f'{DATA_ROOT}/inria_holidays/jpg_small')
    spu.parallel_map_with_progbar(rotate_resize_save, image_paths)
    shutil.rmtree(f'{DATA_ROOT}/inria_holidays/jpg')


def rotate_resize_save(src_path):
    dst_path = src_path.replace('inria_holidays/jpg', 'inria_holidays/jpg_small')
    im = load_image_with_proper_orientation(src_path)
    im = crop_center_square(im)
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    imageio.imwrite(dst_path, im, quality=95)


def load_image_with_proper_orientation(filepath):
    try:
        image = Image.open(filepath)
        orientation_exif_index = 274
        exif_info = dict(image.getexif().items())

        if exif_info[orientation_exif_index] == 3:
            image = image.rotate(180, expand=True)
        elif exif_info[orientation_exif_index] == 6:
            image = image.rotate(270, expand=True)
        elif exif_info[orientation_exif_index] == 8:
            image = image.rotate(90, expand=True)
        return np.asarray(image)
    except (AttributeError, KeyError, IndexError):
        # No EXIF found
        return imageio.imread(filepath)


def crop_center_square(im):
    height, width, channels = im.shape
    if height == width:
        return im
    elif height < width:
        x_start = (width - height) // 2
        im = im[:, x_start:x_start + height, :]
    else:
        y_start = (height - width) // 2
        im = im[y_start:y_start + width, :, :]
    return im


if __name__ == '__main__':
    main()
