import os.path as osp

import cameralib
import numpy as np
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    root = f'{DATA_ROOT}/chico'
    video_paths = spu.sorted_recursive_glob(f'{root}/dataset_raw/*/*.mp4')
    video_relpaths = [osp.relpath(p, root) for p in video_paths]
    cameras = load_cameras()
    all_cameras = {
        p: cameras[get_camera_name(p)] for p in video_relpaths}
    spu.dump_pickle(all_cameras, f'{root}/cameras_only_intrinsics.pkl')


def get_camera_name(video_path):
    return video_path[-9:-4]


def load_cameras():
    cameras = {}
    for c in spu.load_json(f'{DATA_ROOT}/chico/camera_calib_parameters.json')['cameras']:
        intrinsic_matrix = np.array(c['K'], dtype=np.float32)
        d = np.array(c['distCoef'], dtype=np.float32)
        name = c['name']
        cameras[name] = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix, distortion_coeffs=d, world_up=(0, -1, 0))
    return cameras


if __name__ == '__main__':
    main()
