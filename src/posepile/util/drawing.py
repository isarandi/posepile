import cv2
import numpy as np
import simplepyutils as spu
from simplepyutils import logger


def rectangle(im, pt1, pt2, color, thickness):
    cv2.rectangle(im, spu.rounded_int_tuple(pt1), spu.rounded_int_tuple(pt2), color, thickness)


def line(im, p1, p2, *args, **kwargs):
    if np.asarray(p1).shape[-1] != 2 or np.asarray(p2).shape[-1] != 2:
        raise Exception('Wrong dimensionality of point in line drawing')

    try:
        cv2.line(im, spu.rounded_int_tuple(p1), spu.rounded_int_tuple(p2), *args, **kwargs)
    except OverflowError:
        logger.warning('Overflow in spu.rounded_int_tuple!')


def draw_box(im, box, color=(255, 0, 0), thickness=5):
    box = np.array(box)
    rectangle(im, box[:2], box[:2] + box[2:4], color, thickness)


def circle(im, center, radius, *args, **kwargs):
    cv2.circle(im, spu.rounded_int_tuple(center), round(radius), *args, **kwargs)


def fill_polygon(img, pts, color):
    pts = pts.reshape((-1, 1, 2))
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(img, [pts], color)
