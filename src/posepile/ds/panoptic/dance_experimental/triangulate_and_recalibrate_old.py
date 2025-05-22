import functools
import itertools
import multiprocessing

import cv2
import imageio.v2 as imageio
import numpy as np
import scipy.optimize
import simplepyutils as spu
import transforms3d.axangles

import posepile.datasets3d as ds3d
import posepile.ds.panoptic.main as panoptic_main
import posepile.util.geom3d as geom3d
import posepile.util.rigid_alignment as rigid_alignment


def main():
    seq_names = [f'160317_moonbaby{i}' for i in range(1, 4)]
    suffix = 'b470be9'
    # seq_names = [f'150821_dance{i}' for i in range(1, 6)]

    for seq_name, i_ref in itertools.product(seq_names, range(1, 11)):
        do_task(seq_name, suffix, i_ref)


def do_task(seq_name, suffix, i_ref=None):
    joint_info = ds3d.get_joint_info('many_sane')

    for i, j in joint_info.stick_figure_edges:
        print(joint_info.names[i], joint_info.names[j])

    seq_dir = f'/globalwork/datasets/cmu-panoptic/{seq_name}'

    if 'moonbaby' in seq_name:
        camera_names = [f'50_{i:02d}' for i in range(1, 11)]
        camera_type = 'kinect'
    else:
        camera_names = [f'00_{i:02d}' for i in range(30) if i != 23]
        camera_type = 'hd'
    print(camera_names)

    video_paths = [f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}.mp4'
                   for cam_name in camera_names]
    if 'dance' in seq_name:
        cameras = panoptic_main.get_cameras(
            f'{seq_dir}/calibration_{seq_name}_corrected_full.json', camera_names)
    else:
        cameras = panoptic_main.get_cameras(
            f'{seq_dir}/calibration_{seq_name}.json', camera_names)
    cameras = [cameras[n] for n in camera_names]
    sane_bone_lengths = spu.load_pickle('/globalwork/sarandi/data/eccv20_sanebones.pkl')
    videos = [frames_of(video_path) for video_path in video_paths]
    result_poses = []
    poses_root = f'/globalwork/sarandi/data/panoptic_estimate_{suffix}'

    if camera_type == 'kinect':
        sync_data = spu.load_json(
            f'/globalwork/datasets/cmu-panoptic/{seq_name}/ksynctables_{seq_name}.json')
        univ_times = [np.array(sync_data['kinect']['color'][f'KINECTNODE{i}']['univ_time'])
                      for i in range(1, 11)]
        t_starts = [next(x for x in u if x > 0) for u in univ_times]
        t_ends = [next(reversed(list(x for x in u if x > 0))) for u in univ_times]
        reference_times = [
            t for t in univ_times[i_ref]
            if t > 0 and all(start <= t <= end for start, end in zip(t_starts, t_ends))]
        indices = [[np.argmin(np.abs(u - ref_time)) for ref_time in reference_times] for u in
                   univ_times]
    else:
        indices = None

    pose_paths = [f'{poses_root}/{seq_name}/{camera_type}Videos/{camera_type}_{cam_name}_poses.pkl'
                  for cam_name in camera_names]
    camposes_per_cam_frame_id = [spu.load_pickle(p) for p in pose_paths]
    poses_per_cam_frame_id = [
        [[cam.camera_to_world(p) for p in camposes_per_id]
         for camposes_per_id in camposes_per_frame_id]
        for cam, camposes_per_frame_id in zip(cameras, camposes_per_cam_frame_id)
    ]

    if camera_type == 'kinect':
        poses_per_cam_frame_id = [
            pick_indices(seq, i) for seq, i in zip(poses_per_cam_frame_id, indices)]
        videos = [pick_indices(seq, i) for seq, i in zip(videos, indices)]

    poses_per_frame_cam_id = list(zip(*poses_per_cam_frame_id))
    views_per_frame = zip(*videos)
    frames_and_poses_per_t = zip(views_per_frame, poses_per_frame_cam_id)
    # frames_and_poses_per_t = zip(poses_per_frame_cam_id, poses_per_frame_cam_id)

    for i_frame, (frames, poses_per_cam_id) in enumerate(
            spu.progressbar(frames_and_poses_per_t)):
        poses_per_id_cam = match_poses(poses_per_cam_id, min_views=5)
        median_poses_of_this_frame = []

        for poses_per_cam in poses_per_id_cam:
            i_cameras, poses_now = zip(*poses_per_cam)
            cameras_now = [cameras[i] for i in i_cameras]
            median_pose = ransac_triangulate_poses(
                poses_now, cameras_now, sane_bone_lengths, joint_info)
            median_poses_of_this_frame.append(median_pose)
        result_poses.append(median_poses_of_this_frame)

    if camera_type == 'kinect':
        i_start_ref = indices[i_ref][0]
        result_poses = [None] * i_start_ref + result_poses
        spu.dump_pickle(
            result_poses, f'{poses_root}/{seq_name}/{camera_type}_{camera_names[i_ref]}.pkl')
    else:
        spu.dump_pickle(result_poses, f'{poses_root}/{seq_name}.pkl')


def frames_of(video_path):
    with imageio.get_reader(video_path, 'ffmpeg') as reader:
        yield from reader


def ransac_triangulate_poses(poses_per_cam, cameras, sane_bone_lengths, joint_info):
    is_sane = [p is not None and is_pose_sane(p, sane_bone_lengths, joint_info)
               for p in poses_per_cam]
    if not any(is_sane):
        return None

    seen_poses = [p for p, s in zip(poses_per_cam, is_sane) if s]
    seen_cameras = [cam for cam, s in zip(cameras, is_sane) if s]

    # Simple median
    median_pose = compute_median_pose(seen_poses, i_root=joint_info.ids.pelv)
    is_visible = np.array([cam.is_visible(median_pose, [1920, 1080])
                           for p, cam in zip(seen_poses, seen_cameras)])
    median_pose = ransac_triangulate_pose(
        seen_cameras, seen_poses, is_visible, median_pose, 100, 8 / 256)

    if (median_pose is not None and
            not is_pose_sane(median_pose, sane_bone_lengths, joint_info)):
        return None

    return median_pose


def pick_indices(seq, indices):
    it = iter(seq)
    i_current = -1
    current = None
    for i_wanted in indices:
        while i_current < i_wanted:
            current = next(it)
            i_current += 1
        yield current


def triangulate_pose(cameras, poses, default_pose, joint_validity_mask):
    proj_matrices = [c.get_projection_matrix() for c in cameras]

    proj_poses = [pose @ P[:3, :3].T + P[:3, 3] for pose, P in zip(poses, proj_matrices)]
    proj_poses = [pose[:, :2] / pose[:, 2:] for pose in proj_poses]

    triangulated = default_pose.copy()
    for i_joint in range(len(default_pose)):
        if sum(valid[i_joint] for valid in joint_validity_mask) < 3:
            continue

        A = np.concatenate([
            proj_pose[i_joint, :, np.newaxis] @ pr[2:] - pr[:2]
            for pr, proj_pose, valid in zip(proj_matrices, proj_poses, joint_validity_mask)
            if valid[i_joint]], axis=0)
        vh = np.linalg.svd(A, full_matrices=False)[2]
        triangulated[i_joint] = vh[3, :3] / vh[3, 3:]
    return triangulated


def solve_triangulation(
        i_valid_points, indices_among_valids, i_joint, A_all, proj_matrices, proj_poses,
        inlier_thresh, box_sizes):
    A = np.reshape(A_all[indices_among_valids], [4, 4])
    vh = np.linalg.svd(A, full_matrices=False)[2]
    triangulated_point = vh[3, :3] / vh[3, 3:]

    reprojected_points = np.array([
        triangulated_point @ proj_matrices[i, :3, :3].T + proj_matrices[i, :3, 3]
        for i in i_valid_points])
    reprojected_points = reprojected_points[:, :2] / reprojected_points[:, 2:]
    reproj_errors = np.linalg.norm(
        reprojected_points - proj_poses[i_valid_points, i_joint], axis=-1)

    inlier_thresh_abs = inlier_thresh * box_sizes[i_valid_points]
    i_inliers = np.argwhere(reproj_errors < inlier_thresh_abs)[:, 0]
    auc = np.mean(np.maximum(0, 1 - reproj_errors / (inlier_thresh_abs * 1.5)))
    return i_inliers, auc


def ransac_triangulate_pose(
        cameras, poses, is_visible, default_pose, n_trials, inlier_thresh):
    proj_matrices = np.array([c.get_projection_matrix() for c in cameras])
    proj_poses = np.array([pose @ P[:3, :3].T + P[:3, 3] for pose, P in zip(poses, proj_matrices)])
    proj_poses = proj_poses[:, :, :2] / proj_poses[:, :, 2:]
    import boxlib
    boxes = np.array([
        boxlib.expand_to_square(
            boxlib.intersection(boxlib.full(imsize=[1920, 1080]),
                                boxlib.bb_of_points(pose))) for pose in proj_poses])
    box_sizes = boxes[:, 2]

    joint_validity_mask = np.logical_and(
        is_visible, np.logical_and(proj_poses[..., 0] < 1920, proj_poses[..., 1] < 1080))

    triangulated = default_pose.copy()
    for i_joint in range(len(default_pose)):
        i_valid_points = np.argwhere(joint_validity_mask[:, i_joint])[:, 0]
        n_valid_points = len(i_valid_points)
        if n_valid_points < 2:
            return None

        A_all = np.stack([
            (proj_pose[i_joint, :, np.newaxis] @ pr[2:] - pr[:2]) / s
            for s, pr, proj_pose, valid in
            zip(box_sizes, proj_matrices, proj_poses, joint_validity_mask)
            if valid[i_joint]], axis=0)

        ransac_indices = [np.random.choice(
            np.arange(n_valid_points), replace=False, size=2) for _ in range(n_trials)]
        args = [
            (i_valid_points, indices_among_valids, i_joint, A_all, proj_matrices, proj_poses,
             inlier_thresh, box_sizes) for indices_among_valids in ransac_indices]

        inlier_indices_aucs = get_pool().starmap(solve_triangulation, args)
        inlier_indices, aucs = zip(*inlier_indices_aucs)
        best_i_inliers = inlier_indices[np.argmax(aucs)]

        if len(best_i_inliers) < 2:
            return None

        A = np.reshape(A_all[best_i_inliers], [-1, 4])
        vh = np.linalg.svd(A, full_matrices=False)[2]
        triangulated[i_joint] = vh[3, :3] / vh[3, 3:]

    return triangulated


def get_bone_lengths(pose, joint_info):
    return np.array([
        np.linalg.norm(pose[i] - pose[j], axis=-1)
        for i, j in joint_info.stick_figure_edges])


def is_pose_sane(pose, sane_bone_lengths, joint_info):
    bone_lengths = get_bone_lengths(pose, joint_info)
    bone_length_relative = bone_lengths / sane_bone_lengths
    bone_length_diff = np.abs(bone_lengths - sane_bone_lengths)

    with np.errstate(invalid='ignore'):
        relsmall = bone_length_relative < 0.1
        relbig = bone_length_relative > 3
        absdiffbig = bone_length_diff > 300

    insane = np.any(np.logical_and(np.logical_or(relbig, relsmall), absdiffbig))
    return not insane


def rootrel(p, i_root=23):
    return p - p[i_root:i_root + 1]


def match_poses(poses_per_camera, min_views=5):
    def pose_dist(x, y):
        return np.mean(np.linalg.norm(x - y, axis=1))

    groups = []
    for i_cam, poses in enumerate(poses_per_camera):
        matches, unmatched_groups, unmatched_poses = hungarian_match(
            groups, poses, dist_fn=lambda g, p: pose_dist(g[0][1], p), threshold=800)

        for i_group, i_pose, group, pose, dist in matches:
            group.append((i_cam, pose))

        for i_pose, pose in unmatched_poses:
            groups.append([(i_cam, pose)])

    return [g for g in groups if len(g) >= min_views]


def recalibrate_camera_full(poses_world, poses_camera, camera_orig):
    def pose_dist(x, y):
        return np.mean(np.linalg.norm(rigid_alignment.rigid_align(x, y) - y, axis=-1))

    all_poses_world = []
    all_poses_image = []

    for i_frame, (poses1, poses2) in enumerate(zip(poses_world, poses_camera)):
        matches = hungarian_match(poses1, poses2, dist_fn=pose_dist, threshold=400)[0]
        for pose1, pose2, _ in matches:
            all_poses_world.append(pose1)
            all_poses_image.append(camera_orig.camera_to_image(pose2))

    all_points_world = np.concatenate(all_poses_world, axis=0)
    all_points_image = np.concatenate(all_poses_image, axis=0)
    thresh_in_px = 10
    return calibrate_ransac(
        all_points_world, all_points_image, thresh_in_px, camera_orig, 100)


def calibrate_ransac(points3d, points2d, thresh, camera_orig, n_trials=1000):
    n_points = len(points3d)
    best_i_inliers = []
    unskewed_cam = camera_orig.copy()
    unskewed_cam.unskew_pixels()
    new_cam = unskewed_cam.copy()

    def prep(x):
        return x.copy()[np.newaxis].astype(np.float32)

    for i_iter in range(n_trials):
        indices = np.random.choice(np.arange(n_points), replace=False, size=13)
        reproj_error, intrinsic_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            prep(points3d[indices]), prep(points2d[indices]),
            cameraMatrix=unskewed_cam.intrinsic_matrix,
            imageSize=(1920, 1080), distCoeffs=unskewed_cam.distortion_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1
                  | cv2.CALIB_FIX_K2
                  | cv2.CALIB_FIX_K3
                  | cv2.CALIB_FIX_K4
                  | cv2.CALIB_FIX_K5
                  | cv2.CALIB_FIX_K6
                  | cv2.CALIB_FIX_TANGENT_DIST)

        new_cam.intrinsic_matrix = intrinsic_matrix
        new_cam.distortion_coeffs = distCoeffs[:, 0]
        new_cam.R = transforms3d.axangles.axangle2mat(rvecs[0][:, 0],
                                                      np.linalg.norm(rvecs[0][:, 0]))
        new_cam.t = -new_cam.R.T @ tvecs[0][:, 0]
        reproj_points = new_cam.world_to_image(points3d)
        i_inliers = np.argwhere(np.linalg.norm(points2d - reproj_points, axis=-1) < thresh)[:, 0]

        if len(i_inliers) > len(best_i_inliers):
            best_i_inliers = i_inliers

    print('inliers', len(best_i_inliers))
    if len(best_i_inliers) >= 4:
        reproj_error, intrinsic_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
            prep(points3d[best_i_inliers]), prep(points2d[best_i_inliers]),
            cameraMatrix=unskewed_cam.intrinsic_matrix,
            imageSize=(1920, 1080), distCoeffs=unskewed_cam.distortion_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1
                  | cv2.CALIB_FIX_K2
                  | cv2.CALIB_FIX_K3
                  | cv2.CALIB_FIX_K4
                  | cv2.CALIB_FIX_K5
                  | cv2.CALIB_FIX_K6
                  | cv2.CALIB_FIX_TANGENT_DIST)

        new_cam.intrinsic_matrix = intrinsic_matrix
        new_cam.distortion_coeffs = distCoeffs[:, 0]
        new_cam.R = transforms3d.axangles.axangle2mat(
            rvecs[0][:, 0],
            np.linalg.norm(rvecs[0][:, 0]))
        new_cam.t = -new_cam.R.T @ tvecs[0][:, 0]
        return new_cam
    else:
        return None


def recalibrate_cameras():
    seq_name = '150821_dance1'
    seq_dir = f'/globalwork/datasets/cmu-panoptic/{seq_name}'
    camera_names = [f'00_{i:02d}' for i in [5, 14, 22, 26, 27, 28, 29]]
    camera_type = 'hd'
    cameras = panoptic_main.get_cameras(
        f'{seq_dir}/calibration_{seq_name}.json', camera_names)
    cameras = [cameras[n] for n in camera_names]

    suffix = 'b470be9'
    poses_root = f'/globalwork/sarandi/data/panoptic_estimate_{suffix}'
    pose_paths = [f'{poses_root}/{seq_name}/{camera_type}Videos/{camera_type}_{cam_name}_poses.pkl'
                  for cam_name in camera_names]

    camposes_per_cam_frame_id = [spu.load_pickle(p) for p in pose_paths]
    world_poses_per_frame_id = spu.load_pickle(f'{poses_root}/{seq_name}.pkl')
    world_poses_per_frame_id = [[x for x in p if x is not None] for p in world_poses_per_frame_id]

    new_cams = {}
    for camposes_per_frame_id, cam, name in zip(camposes_per_cam_frame_id, cameras, camera_names):
        new_cam = recalibrate_camera_full(world_poses_per_frame_id, camposes_per_frame_id, cam)
        print(cam.R, new_cam.R)
        print(cam.t, new_cam.t)
        print(cam.distortion_coeffs, new_cam.distortion_coeffs)
        print(cam.intrinsic_matrix, new_cam.intrinsic_matrix)
        print('--')
        new_cams[name] = new_cam

    cameras_json = spu.load_json(f'{seq_dir}/calibration_{seq_name}.json')
    for c in cameras_json['cameras']:
        if c['name'] in camera_names:
            new_cam = new_cams[c['name']]
            t = (-new_cam.R @ new_cam.t).reshape([3, 1]) / 10
            c['R'] = new_cam.R.tolist()
            c['t'] = t.tolist()
            c['K'] = new_cam.intrinsic_matrix.tolist()
            c['distCoef'] = new_cam.distortion_coeffs.tolist()
    spu.dump_json(cameras_json, f'{seq_dir}/calibration_{seq_name}_corrected_full.json',
                  indent=4, sort_keys=True)


def compute_median_pose(poses, joint_validity_mask=None, eps=5, i_root=-1):
    poses = np.asarray(poses)
    rootrel_poses = poses - poses[:, i_root][:, np.newaxis]

    if joint_validity_mask is None:
        joint_validity_mask = np.full(poses.shape[:2], True)

    rootrel_median = np.stack([
        masked_median(rootrel_poses[:, i], joint_validity_mask[:, i], eps=eps)
        for i in range(rootrel_poses.shape[1])])

    root_median = masked_median(poses[:, i_root], joint_validity_mask[:, i_root], eps=eps)
    return root_median + rootrel_median


def masked_median(coords, mask, eps=5):
    valid_coords = coords[mask]
    if len(valid_coords) > 0:
        return geom3d.geometric_median(valid_coords, eps=eps)
    else:
        return geom3d.geometric_median(coords, eps=eps)


def hungarian_match(objs1, objs2, *, dist_fn=None, sim_fn=None, threshold=np.inf):
    if len(objs1) == 0 or len(objs2) == 0:
        return []
    if (sim_fn is None) == (dist_fn is None):
        raise Exception('Exactly one of dist_fn or sim_fn must be given.')

    if dist_fn is None:
        dist_fn = lambda x, y: -sim_fn(x, y)
        threshold = -threshold

    distance_matrix = np.array([[dist_fn(obj1, obj2) for obj2 in objs2] for obj1 in objs1])
    matched_indices1, matched_indices2 = scipy.optimize.linear_sum_assignment(distance_matrix)
    matches = [
        (i1, i2, objs1[i1], objs2[i2], distance_matrix[i1, i2])
        for i1, i2 in zip(matched_indices1, matched_indices2)
        if distance_matrix[i1, i2] < threshold]

    unmatched_objs1 = [(i, o) for i, o in enumerate(objs1) if i not in matched_indices1]
    unmatched_objs2 = [(i, o) for i, o in enumerate(objs2) if i not in matched_indices2]
    return matches, unmatched_objs1, unmatched_objs2


@functools.lru_cache()
def get_pool():
    return multiprocessing.Pool(12)


if __name__ == '__main__':
    # recalibrate_cameras()
    main()
