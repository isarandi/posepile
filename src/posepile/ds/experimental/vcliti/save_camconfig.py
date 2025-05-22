import glob
import os
import os.path as osp

import cameralib
import numpy as np
import simplepyutils as spu

from posepile.paths import DATA_ROOT
from posepile.ds.experimental.vcliti.main import VCLITI_ROOT


def main():
    calib_paths = spu.sorted_recursive_glob(f'{VCLITI_ROOT}/**/*calibration.txt')
    seq_paths = sorted(set(['/'.join(spu.split_path(p)[:-3]) for p in calib_paths]))
    seq_relpaths = [osp.relpath(p, VCLITI_ROOT) for p in seq_paths]

    frame_relpath_to_camera = {}
    for seq_relpath in seq_relpaths:
        video_dirs = glob.glob(f'{VCLITI_ROOT}/{seq_relpath}/D?')
        cam_ids = [v[-1] for v in video_dirs]
        cameras = [load_camera(seq_relpath, i)[0] for i in cam_ids]

        frame_seqs = [glob.glob(f'{VCLITI_ROOT}/{seq_relpath}/D{cam_id}/*.jpg') for cam_id in
                      cam_ids]
        for frame_seq, camera in zip(frame_seqs, cameras):
            for frame_path in frame_seq:
                frame_relpath = osp.relpath(frame_path, VCLITI_ROOT)
                frame_relpath_to_camera[frame_relpath] = camera

    spu.dump_pickle(frame_relpath_to_camera, f'{VCLITI_ROOT}/cameras.pkl')


def load_camera(seq_relpath, cam_id):
    calib_dir = f'{VCLITI_ROOT}/{seq_relpath}/calibration'
    try:
        lines = spu.read_lines(f'{calib_dir}/intrinsics/KRT{cam_id}.txt')
    except:
        # Typo in original dataset
        lines = spu.read_lines(f'{calib_dir}/intinsics/KRT{cam_id}.txt')

    K = np.array([float(x) for x in lines[1].split()]).reshape(3, 3)
    R = np.array([float(x) for x in lines[2].split()]).reshape(3, 3)
    t = np.array([float(x) for x in lines[3].split()])
    depth_to_rgb = get_extrinsics(R, t)

    lines = spu.read_lines(f'{calib_dir}/extrinsics/D{cam_id}_calibration.txt')
    R = np.array([[float(x) for x in line.split()] for line in lines[:3]])
    t = np.array([float(x) for x in lines[3].split()])
    depth_to_world = get_extrinsics(R, t)

    world_to_depth = np.linalg.inv(depth_to_world)
    world_to_rgb = depth_to_rgb @ world_to_depth
    world_up = (0, 0, 1) if seq_relpath.startswith('Skeleton-dataset') else (0, 1, 0)
    cam_rgb = cameralib.Camera(
        intrinsic_matrix=K, extrinsic_matrix=world_to_rgb, world_up=world_up)
    cam_depth = cameralib.Camera(
        intrinsic_matrix=K, extrinsic_matrix=world_to_depth, world_up=world_up)
    cam_rgb.horizontal_flip_image([1080, 1920])
    cam_depth.horizontal_flip_image([424, 512])
    return cam_rgb, cam_depth


def get_extrinsics(R, t):
    result = np.eye(4)
    result[:3, :3] = R.reshape(3, 3)
    result[:3, 3] = t.reshape(3)
    return result


if __name__ == '__main__':
    main()
