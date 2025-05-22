import cv2
import numpy as np
import scipy.optimize
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    filepaths2d = spu.sorted_recursive_glob(f'{DATA_ROOT}/sailvos/**/*_2d.txt')
    spu.parallel_map_with_progbar(process, filepaths2d)


def process(filepath2d):
    save_path = filepath2d.replace('_2d.txt', '_camera.npz')
    # if osp.exists(save_path):
    #     return
    coords2d = np.loadtxt(filepath2d, dtype=np.float32)
    coords3d = np.loadtxt(filepath2d.replace('_2d.txt', '_3d.txt'), dtype=np.float32)
    intrinsic_matrix, extrinsic_matrix = calibrate_camera(coords2d, coords3d, (1280, 800))
    if intrinsic_matrix is None:
        np.savez(save_path, intrinsics=np.nan, extrinsics=np.nan)
    else:
        np.savez(save_path, intrinsics=intrinsic_matrix, extrinsics=extrinsic_matrix)


def calibrate_camera(image_coords2d, world_coords3d, imsize):
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS
             | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4
             | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 | cv2.CALIB_FIX_TANGENT_DIST)
    is_valid3d = np.any(world_coords3d != 0, axis=-1)
    is_valid2d = np.logical_and(image_coords2d[:, 0] != -1280, image_coords2d[:, 0] != -800)
    is_valid = np.logical_and(is_valid3d, is_valid2d)

    # DLT algorithm needs at least 6 points for camera pose estimation
    # from 3D-2D point correspondences.
    if np.count_nonzero(is_valid) < 6:
        return None, None

    coords2d = image_coords2d[np.newaxis, is_valid]
    coords3d = world_coords3d[np.newaxis, is_valid]

    # Subtracting the mean helps with numerical stability
    mean3d = np.mean(coords3d, axis=1, keepdims=True)
    coords3d = coords3d - mean3d

    def estimate_camera(hfov_degrees):
        hfov_radians = np.deg2rad(hfov_degrees)
        initial_focal = imsize[0] / (2 * np.tan(hfov_radians / 2))
        initial_intrinsic_matrix = np.eye(3, dtype=np.float32)
        initial_intrinsic_matrix[0, 0] = initial_focal
        initial_intrinsic_matrix[1, 1] = initial_focal
        initial_intrinsic_matrix[:2, 2] = np.array(imsize) / 2
        return cv2.calibrateCamera(
            coords3d, coords2d, cameraMatrix=initial_intrinsic_matrix, imageSize=imsize,
            distCoeffs=None, flags=flags)

    optimal_hfov_degrees = scipy.optimize.minimize_scalar(
        lambda x: estimate_camera(x)[0], bounds=(30, 100), method='bounded').x
    reproj_error, intrinsic_matrix, distCoeffs, rvecs, tvecs = estimate_camera(optimal_hfov_degrees)
    rot_matrix = cv2.Rodrigues(rvecs[0])[0]

    t = tvecs[0] - rot_matrix @ mean3d.reshape(3, 1)
    extrinsic_matrix = np.concatenate([rot_matrix, t], axis=1)
    return intrinsic_matrix, extrinsic_matrix


if __name__ == '__main__':
    main()
