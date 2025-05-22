import functools

import cv2
import numpy as np
import rlemasklib


@functools.lru_cache()
def get_structuring_element(shape, ksize, anchor=None):
    if not isinstance(ksize, tuple):
        ksize = (ksize, ksize)
    return cv2.getStructuringElement(shape, ksize, anchor)


def largest_connected_component(mask):
    mask = mask.astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    areas = stats[1:, -1]
    if len(areas) < 1:
        return mask, np.zeros(4, np.float32)

    largest_area_label = 1 + np.argsort(areas)[-1]
    obj_mask = np.uint8(labels == largest_area_label)
    obj_box = stats[largest_area_label, :4]

    return obj_mask, np.asarray(obj_box, np.float32)


def mask_iou(mask1, mask2):
    intersection = cv2.countNonZero(mask1 & mask2)
    if intersection == 0:
        return 0
    union = cv2.countNonZero(mask1 | mask2)
    return intersection / union


def erode(mask, kernel_size, iterations=1):
    elem = get_structuring_element(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_ERODE, elem, iterations=iterations)


def dilate(mask, kernel_size, iterations=1):
    elem = get_structuring_element(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, elem, iterations=iterations)


def masks_to_label_map(masks):
    h, w = masks.shape[1:3]
    final_mask = np.zeros([h, w], np.uint8)
    i_instance = 1
    for mask in masks:
        final_mask[mask > 0.5] = i_instance
        i_instance += 1
    return final_mask


def resize_mask(mask_encoded, new_imshape):
    mask = rlemasklib.decode(mask_encoded) * 255
    mask = cv2.resize(mask, (new_imshape[1], new_imshape[0]))
    mask //= 128
    return rlemasklib.encode(mask)


def draw_mask(img, mask, mask_color, gap=1, thickness=5):
    bool_mask = mask != 0
    if not np.any(bool_mask):
        return

    inline = get_inline(bool_mask, gap, thickness)
    imcolor = img[bool_mask].astype(np.float64)
    mask_color = np.asarray(mask_color, np.float64)
    img[bool_mask] = np.clip(
        mask_color * 0.3 + imcolor * 0.7, 0, 255).astype(np.uint8)
    if np.any(inline):
        img[inline] = mask_color


def get_inline(mask, d1=1, d2=3):
    if mask.dtype == bool:
        return get_inline(mask.astype(np.uint8), d1, d2).astype(bool)
    return (erode(mask, d1) if d1 > 0 else mask )- erode(mask, d2)
