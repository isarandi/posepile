import itertools
import os
import os.path as osp

import numpy as np
import simplepyutils as spu
from posepile.ds.experimental.cwi.triangulate import find_main_person, triangulate_poses
from posepile.ds.experimental.triangulate_common import mask_and_average
from posepile.ds.experimental.vcliti.main import VCLITI_ROOT
from posepile.ds.experimental.vcliti.save_camconfig import load_camera
from posepile.paths import DATA_ROOT


def main():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    preds_all = spu.load_pickle(f'{VCLITI_ROOT}/metrabs_pred.pkl')
    joint_info = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')

    calib_paths = spu.sorted_recursive_glob(f'{VCLITI_ROOT}/**/*calibration.txt')
    seq_paths = sorted(set([spu.path_range(p, 0, -3) for p in calib_paths]))
    seq_relpaths = [osp.relpath(p, VCLITI_ROOT) for p in seq_paths]
    seq_relpath = seq_relpaths[i_task]
    seq_path = f'{VCLITI_ROOT}/{seq_relpath}'
    out_path = f'{VCLITI_ROOT}/triang/{seq_relpath}/output.pkl'
    if spu.is_pickle_readable(out_path):
        return
    cam_offsets = load_offsets(f'{seq_path}/synchronization/synchronization.txt')

    camera_names = sorted(cam_offsets.keys())
    cameras = [load_camera(seq_relpath, camera_name[-1])[0] for camera_name in camera_names]
    offset_per_cam = [cam_offsets[camera_name] for camera_name in camera_names]
    timestamps_per_cam = [
        np.load(f'{seq_path}/{cam_name}_timestamps.npz')['rgb_timestamps']
        for cam_name in camera_names]
    correspondence = select_corresponding_indices(offset_per_cam, timestamps_per_cam)
    indices_per_cam = list(zip(*correspondence))

    frame_paths_per_cam = [
        spu.sorted_recursive_glob(f'{seq_path}/{cam_name}/*.jpg')
        for cam_name in camera_names]
    frame_relpaths_per_cam = [
        [osp.relpath(p, VCLITI_ROOT) for p in paths]
        for paths in frame_paths_per_cam]
    poses_per_cam = [
        [[p for p, b in zip(preds_all[path]['poses3d'], preds_all[path]['boxes'])
          if b[-1] > 0.5]
         for path in paths]
        for paths in frame_relpaths_per_cam]
    poses_corresp_per_cam = [
        [poses[i] for i in indices]
        for poses, indices in zip(poses_per_cam, indices_per_cam)]

    triangs = []
    for world_poses_per_cam in spu.progressbar(
            zip(*poses_corresp_per_cam), total=len(poses_corresp_per_cam[0])):
        world_pose_per_cam = find_main_person(world_poses_per_cam, joint_info)
        campose_per_cam = np.array([
            c.world_to_camera(p) for c, p in zip(cameras, world_pose_per_cam)])
        campose_per_cam = mask_and_average(campose_per_cam)
        triang = triangulate_poses(cameras, campose_per_cam)
        triangs.append(triang)

    triangs = np.array(triangs)
    spu.dump_pickle(dict(triangs=triangs, indices_per_cam=indices_per_cam), out_path)


def select_corresponding_indices(offset_per_cam, timestamps_per_cam):
    adjusted_timestamps_per_cam = [
        [np.float64(stamp) - np.float64(offset) for stamp in stamps]
        for stamps, offset in zip(timestamps_per_cam, offset_per_cam)]
    n_cameras = len(offset_per_cam)
    corresp = [[0] * n_cameras]
    candidate_moves = list(itertools.product(*[(0, 1)] * n_cameras))[1:]
    current_indices = [0] * n_cameras
    while True:
        candidate_indices = [
            [i + m for i, m in zip(current_indices, move)] for move in candidate_moves]
        try:
            candidate_stamps = [
                [t[i] for i, t in zip(inds, adjusted_timestamps_per_cam)]
                for inds in candidate_indices]
        except IndexError:
            break

        max_absdiff_per_move = [max_absdiff(stamps) for stamps in candidate_stamps]
        i_best = np.argmin(max_absdiff_per_move)
        best_absdiff = max_absdiff_per_move[i_best]
        current_indices = candidate_indices[i_best]
        if best_absdiff <= 200000:
            corresp.append(current_indices)
    return corresp


def max_absdiff(timestamps):
    return max(abs(t1 - t2) for t1 in timestamps for t2 in timestamps)


def load_offsets(path):
    line_parts = [l.split() for l in spu.read_lines(path)]
    return {parts[0]: parts[2] for parts in line_parts if parts and parts[0][0] == 'D'}


if __name__ == '__main__':
    main()
