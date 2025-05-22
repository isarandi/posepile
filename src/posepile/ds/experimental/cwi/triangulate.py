import glob
import itertools
import os
import os.path as osp

import boxlib
import numpy as np
import simplepyutils as spu

from posepile.ds.experimental.triangulate_common import infargmin, infmin, mask_and_average, project
from posepile.paths import DATA_ROOT
from posepile.util import geom3d
import random
CWI_ROOT = f'{DATA_ROOT}/cwi'


def main():
    video_paths_all = sorted(glob.glob(f'{CWI_ROOT}/*/*/raw_files/*.mkv'))
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    video_paths = list(spu.groupby(video_paths_all, osp.dirname).values())[i_task]
    print(video_paths)
    video_relpaths = [osp.relpath(p, CWI_ROOT) for p in video_paths]

    out_path = f'{CWI_ROOT}/triang2/{osp.dirname(video_relpaths[0])}/output.pkl'
    # if spu.is_pickle_readable(out_path):
    #    return

    cameras_all = spu.load_pickle(f'{CWI_ROOT}/cameras.pkl')
    cameras = [cameras_all[p] for p in video_relpaths]
    preds = [spu.load_pickle(spu.replace_extension(f'{CWI_ROOT}/pred/{p}', '.pkl'))
             for p in video_relpaths]
    poses3d = [p['poses3d'] for p in preds]

    joint_info = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')

    triangs = []
    for world_poses_per_cam in spu.progressbar(zip(*poses3d), total=len(poses3d[0])):
        world_pose_per_cam = find_main_person(world_poses_per_cam, joint_info)
        campose_per_cam = np.array([
            c.world_to_camera(p) for c, p in zip(cameras, world_pose_per_cam)])
        campose_per_cam = mask_and_average(campose_per_cam)
        triang = triangulate_poses(cameras, campose_per_cam)
        triangs.append(triang)

    triangs = np.array(triangs)
    spu.dump_pickle(triangs, out_path)


def find_main_person(poses_per_cam, joint_info, n_aug=5, root_name='pelv', distance_thresh=400):
    i_pelv = joint_info.ids[root_name]
    pelvises_per_cam = [np.mean(poses, -3)[:, i_pelv] for poses in poses_per_cam]  # cam, person, 3

    candidates = np.concatenate(pelvises_per_cam, axis=0)  # cam_person, 3
    if len(candidates) == 0:
        return np.full(
            [len(poses_per_cam), n_aug, joint_info.n_joints, 3], dtype=np.float32,
            fill_value=np.nan)

    n_match_per_cam = np.count_nonzero([
        np.any(
            np.linalg.norm(pelvises[..., np.newaxis, :] - candidates, axis=-1) < distance_thresh,
            axis=0) for pelvises in pelvises_per_cam], axis=0)

    best_candidates = candidates[n_match_per_cam == np.max(n_match_per_cam)]
    ref_point = np.mean(best_candidates, axis=0)
    distances_per_cam = [
        np.linalg.norm(pelvises - ref_point, axis=-1)
        for pelvises in pelvises_per_cam]

    nan = np.full([n_aug, joint_info.n_joints, 3], dtype=np.float32, fill_value=np.nan)
    best_pose_per_cam = [
        (poses[infargmin(distances)] if infmin(distances) < distance_thresh else nan)
        for poses, distances in zip(poses_per_cam, distances_per_cam)]
    return np.array(best_pose_per_cam)


def triangulate_poses(
        cameras, cam_poses_per_cam, imshape=(1536, 2048), use_triple_combinations=True,
        min_inlier_views=3, max_combos=None, inlier_threshold=0.2, inlier_auc_threshold=0.1):
    proj_poses = [project(p) for c, p in zip(cameras, cam_poses_per_cam)]
    box_sizes = [bounding_box_size(p, imshape=imshape) for p in proj_poses]
    n_joints = len(proj_poses[0])
    valid_cam_indices = [i for i, s in enumerate(box_sizes) if s > 0]
    if len(valid_cam_indices) < min_inlier_views:
        return np.full(shape=[n_joints, 3], dtype=np.float32, fill_value=np.nan)

    cameras = [cameras[i] for i in valid_cam_indices]
    cam_poses_per_cam = [cam_poses_per_cam[i] for i in valid_cam_indices]
    proj_poses = [proj_poses[i] for i in valid_cam_indices]
    box_sizes = [box_sizes[i] for i in valid_cam_indices]
    proj_matrices = [c.get_extrinsic_matrix()[:3] for c in cameras]

    A = np.array([
        [(proj_pose[i_joint, :, np.newaxis] @ pr[2:] - pr[:2]) / s
         for s, pr, proj_pose in zip(box_sizes, proj_matrices, proj_poses)]
        for i_joint in range(n_joints)])  # n_joints, n_views, 2, 4

    # A guess is the triangulated result from 2 or 3 views. We perform all combinations.
    # Like RANSAC without the randomness (just SAC).
    n_views = len(cameras)
    view_combinations = itertools.combinations(range(n_views), 2)
    combos = list(view_combinations)
    if max_combos:
        combos = random.sample(combos, min(len(combos), max_combos))
    guesses2 = reshaped_nullspace(np.stack([A[:, list(i_views)] for i_views in combos]))

    if use_triple_combinations and n_views >= 3:
        view_combinations = itertools.combinations(range(n_views), 3)
        combos = list(view_combinations)
        if max_combos:
            combos = random.sample(combos, min(len(combos), max_combos))
        guesses3 = reshaped_nullspace(
            np.stack([A[:, list(i_views)] for i_views in combos]))
        guesses = np.concatenate([guesses2, guesses3], axis=0)
    else:
        guesses = guesses2

    proj_guesses = [
        [project_pose(proj_mat, guess) for proj_mat in proj_matrices]
        for guess in guesses]  # n_guess, n_views, n_joints, 2
    rel_errors = np.array([[np.linalg.norm(pg - proj_pose, axis=-1) / s
                            for pg, proj_pose, s in zip(proj_guess, proj_poses, box_sizes)]
                           for proj_guess in proj_guesses])  # n_guess, n_views, n_joints

    rel_errors = np.nan_to_num(rel_errors, nan=np.inf)
    is_inlier = rel_errors < inlier_threshold  # n_guess, n_views, n_joints
    # n_inliers = np.count_nonzero(is_inlier, axis=1)  # n_guess, n_joints
    auc_inliers = np.mean(geom3d.auc(rel_errors, 0, inlier_auc_threshold), axis=1)
    # median_err_over_cameras = kth_smallest(rel_errors, 1, axis=1)
    # best_i_guess = infargmin(median_err_over_cameras, axis=0)
    best_i_guess = np.argmax(auc_inliers, axis=0)  # n_joints

    result = np.empty_like(cam_poses_per_cam[0])
    for i_joint in range(n_joints):
        inlier_mask = is_inlier[best_i_guess[i_joint], :, i_joint]
        if np.count_nonzero(inlier_mask) < min_inlier_views:
            result[i_joint] = np.nan
        else:
            result[i_joint] = reshaped_nullspace(A[i_joint, inlier_mask])

    return result


def kth_smallest(arr, k, axis):
    partitioned = np.partition(arr, k, axis=axis)
    sl = tuple([(k if a == axis else slice(None)) for a in range(arr.ndim)])
    return partitioned[tuple(sl)]


def reshaped_nullspace(A):
    A_reshaped = A.reshape([-1, A.shape[-3] * A.shape[-2], A.shape[-1]])
    result_reshaped = nullspace_vector_nansafe(A_reshaped)
    return result_reshaped.reshape([*A.shape[:-3], 3])


def nullspace_vector_nansafe(A):
    is_valid = np.all(np.isfinite(A), axis=(1, 2))
    result_valid = nullspace_vector(A[is_valid])
    result = np.full(shape=[A.shape[0], 3], dtype=np.float32, fill_value=np.nan)
    result[is_valid] = result_valid
    return result


def nullspace_vector(A):
    vh = np.linalg.svd(A, full_matrices=False)[2]
    return vh[..., 3, :3] / vh[..., 3, 3:]


def project_pose(proj_mat, p):
    p_hom = p @ proj_mat[:3, :3].T + proj_mat[:3, 3]
    return p_hom[:, :2] / p_hom[:, 2:]


def bounding_box_size(pose, imshape):
    x, y, width, height = boxlib.intersection(
        boxlib.full(imshape=imshape),
        boxlib.bb_of_points(pose))
    return max(width, height)


if __name__ == '__main__':
    main()
