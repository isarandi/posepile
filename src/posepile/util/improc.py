import functools
import os
import os.path as osp
from fractions import Fraction

import PIL.Image
import cv2
import imageio.v2 as imageio
import numba
import numpy as np
import simplepyutils as spu
from simplepyutils import logger

from posepile import util

VALID_SCALES = [
    Fraction(2, 1), Fraction(15, 8), Fraction(7, 4), Fraction(13, 8),
    Fraction(3, 2), Fraction(11, 8), Fraction(5, 4), Fraction(9, 8),
    Fraction(1, 1), Fraction(7, 8), Fraction(3, 4), Fraction(5, 8),
    Fraction(1, 2), Fraction(3, 8), Fraction(1, 4), Fraction(1, 8)]

use_libjpeg_turbo = True
if use_libjpeg_turbo:
    import jpeg4py


    def _imread(path, *, dst=None, scale=1):
        if not path.lower().endswith(('.jpg', '.jpeg')):
            if scale != 1:
                raise ValueError('downscale_factor only supported for JPEG images')
            return cv2.imread(path)[..., ::-1]
        try:
            return decode_jpeg(path, dst=dst, scale=scale)
        except jpeg4py.JPEGRuntimeError:
            logger.error(f'Could not load image at {path}, JPEG error.')
            raise
else:
    def _imread(path, dst=None, downscale_factor=1):
        assert dst is None
        if downscale_factor != 1:
            raise NotImplementedError('downscale_factor only implemented for libjpeg-turbo')
        return imageio.imread(path)


def imread(path, *, dst=None, scale=1):
    if isinstance(path, bytes):
        path = path.decode('utf8')
    path = util.ensure_absolute_path(path)
    return _imread(path, dst=dst, scale=scale)[..., :3]


def decode_jpeg(data, *, dst=None, scale=None):
    j = make_jpeg(data)
    if dst is None:
        if j.width is None:
            j.parse_header()
        w, h = get_scaled_size(j.width, j.height, scale)
        dst = np.empty([h, w, 3], np.uint8)
    return j.decode(dst)


def make_jpeg(data):
    if isinstance(data, (str, np.ndarray)):
        return jpeg4py.JPEG(data)
    elif isinstance(data, jpeg4py.JPEG):
        return data
    elif isinstance(data, bytes):
        return jpeg4py.JPEG(np.frombuffer(data, np.uint8))
    else:
        raise TypeError(f'Unsupported data type {type(data)} for JPEG')


def get_scaled_size(width, height, scale=1):
    scale = Fraction.from_float(scale)
    if scale not in VALID_SCALES:
        raise ValueError("scale should be one of", list(map(str, VALID_SCALES)))
    return (
        (width * scale.numerator + scale.denominator - 1) // scale.denominator,
        (height * scale.numerator + scale.denominator - 1) // scale.denominator)


def decode_imsize_jpeg(data):
    j = make_jpeg(data)
    j.parse_header()
    return j.width, j.height


def resize_by_factor(im, factor, interp=None):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = spu.rounded_int_tuple([im.shape[1] * factor, im.shape[0] * factor])
    if interp is None:
        interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def image_extents(filepath):
    """Returns the image (width, height) as a numpy array, without loading the pixel data."""

    with PIL.Image.open(filepath) as im:
        return np.asarray(im.size)


def normalize01(im, dst=None):
    if dst is None:
        result = np.empty_like(im, dtype=np.float32)
    else:
        result = dst
    result[:] = im.astype(np.float32) / np.float32(255)
    np.clip(result, np.float32(0), np.float32(1), out=result)
    return result


@numba.jit(nopython=True)
def paste_over(im_src, im_dst, alpha, center, inplace=False):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending.

    The resulting image has the same shape as `im_dst` but contains `im_src`
    (perhaps only partially, if it's put near the border).
    Locations outside the bounds of `im_dst` are handled as expected
    (only a part or none of `im_src` becomes visible).

    Args:
        im_src: The image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) image of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
        inplace: directly change im_dst in place, instead of copying it before modifying

    Returns:
        An image of the same shape as `im_dst`, with `im_src` pasted onto it.
    """

    width_height_src = np.array([im_src.shape[1], im_src.shape[0]], dtype=np.int32)
    width_height_dst = np.array([im_dst.shape[1], im_dst.shape[0]], dtype=np.int32)

    center_float = center.astype(np.float32)
    np.round(center_float, 0, center_float)
    center_int = center_float.astype(np.int32)
    ideal_start_dst = center_int - width_height_src // np.int32(2)
    ideal_end_dst = ideal_start_dst + width_height_src

    zeros = np.zeros_like(ideal_start_dst)
    start_dst = np.minimum(np.maximum(ideal_start_dst, zeros), width_height_dst)
    end_dst = np.minimum(np.maximum(ideal_end_dst, zeros), width_height_dst)

    if inplace:
        result = im_dst
    else:
        result = im_dst.copy()

    region_dst = result[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - ideal_start_dst
    end_src = width_height_src + (end_dst - ideal_end_dst)

    alpha_expanded = np.expand_dims(alpha, -1)
    alpha_expanded = alpha_expanded[start_src[1]:end_src[1], start_src[0]:end_src[0]]

    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]

    result[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
        (alpha_expanded * region_src + (1 - alpha_expanded) * region_dst)).astype(np.uint8)
    return result


def adjust_gamma(image, gamma, inplace=False):
    if inplace:
        cv2.LUT(image, _get_gamma_lookup_table(gamma), dst=image)
        return image

    return cv2.LUT(image, _get_gamma_lookup_table(gamma))


@functools.lru_cache()
def _get_gamma_lookup_table(gamma):
    return (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)


def blend_image(im1, im2, im2_weight):
    if im2_weight.ndim == im1.ndim - 1:
        im2_weight = im2_weight[..., np.newaxis]

    return _blend_image_numba(
        im1.astype(np.float32),
        im2.astype(np.float32),
        im2_weight.astype(np.float32)).astype(im1.dtype)


@numba.jit(nopython=True)
def _blend_image_numba(im1, im2, im2_weight):
    return im1 * (1 - im2_weight) + im2 * im2_weight


def is_image_readable(path):
    try:
        imread(path)
        return True
    except Exception:
        return False


def imwrite_jpeg(image, path, quality=95):
    spu.ensure_parent_dir_exists(path)
    try:
        cv2.imwrite(path, image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    except BaseException as e:
        # clean up the file if it was created, as it will otherwise be corrupted
        if osp.isfile(path):
            os.remove(path)
        raise


def is_jpeg_readable(path):
    try:
        jpeg4py.JPEG(path).decode()
        return True
    except jpeg4py.JPEGRuntimeError:
        return False


def white_balance(img, a=None, b=None):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = a if a is not None else np.mean(result[..., 1])
    avg_b = b if b is not None else np.mean(result[..., 2])
    result[..., 1] = result[..., 1] - ((avg_a - 128) * (result[..., 0] / 255.0) * 1.1)
    result[..., 2] = result[..., 2] - ((avg_b - 128) * (result[..., 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result
