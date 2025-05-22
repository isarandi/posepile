import cv2
import imageio.v2 as imageio
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    pgm_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/bmhad/Camera/**/*.pgm')
    spu.parallel_map_with_progbar(convert_image, pgm_paths)


def convert_image(in_path):
    im_bayer = cv2.imread(in_path, cv2.IMREAD_ANYDEPTH)
    im_rgb = cv2.cvtColor(im_bayer, cv2.COLOR_BAYER_GB2RGB)
    out_path = in_path.replace('pgm', 'jpg')
    imageio.imwrite(out_path, im_rgb, quality=92)


if __name__ == '__main__':
    main()
