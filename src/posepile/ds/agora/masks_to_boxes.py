import re

import cv2
import numpy as np
import rlemasklib
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    image_filenames = spu.sorted_recursive_glob(f'{DATA_ROOT}/agora/**/.png')
    mask_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/agora/**/*_mask_*.pkl')

    with spu.ThrottledPool() as pool:
        for mask_path in spu.progressbar(mask_paths):
            pool.apply_async(save_mask, (mask_path,))


def extract_mask(mask_path):
    modal_masks = []
    amodal_masks = []

    label_colors = label_colors[..., ::-1]  # OpenCV loads as BGR, so swap the order of label colors
    label_im = cv2.imread(all_mask_paths[0]).astype(np.int32)
    for color, amodal_mask_path in zip(label_colors, all_mask_paths[1:]):
        amodal_mask_im = cv2.imread(amodal_mask_path)
        amodal_mask = rlemasklib.encode(np.any(amodal_mask_im != 0, axis=-1))
        modal_mask = rlemasklib.encode(np.all(np.abs(label_im - color) < 10, axis=-1))
        modal_mask = rlemasklib.intersection([modal_mask, amodal_mask])
        amodal_masks.append(amodal_mask)
        modal_masks.append(modal_mask)

    overall_mask = rlemasklib.encode(
        np.logical_not(np.all(np.equal(label_im, [0, 0, 255]), axis=-1)))
    spu.dump_pickle(
        data=dict(all_people=overall_mask, modal=modal_masks, amodal=amodal_masks),
        file_path=all_mask_paths[0].replace('.png', '.pkl'))


def load_colors(path):
    lines = spu.read_lines(path)
    pattern = r'\d+\t\[(?P<r>\d+), (?P<g>\d+), (?P<b>\d+)\]'
    matches = [re.match(pattern, l) for l in lines]
    colors = [(int(m['r']), int(m['g']), int(m['b'])) for m in matches]
    return np.array(colors, np.int32)


if __name__ == '__main__':
    main()
