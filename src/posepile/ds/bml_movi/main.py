import argparse
import glob
import itertools
import os
import os.path as osp
import re

import boxlib
import cameralib
import cv2
import einops
import imageio.v2 as imageio
import numpy as np
import scipy.optimize
import simplepyutils as spu
from simplepyutils import FLAGS, logger

import posepile.datasets3d as ds3d
import posepile.ds.bml_movi.rigid_alignment as rigid_alignment
import posepile.util.geom3d as geom3d
import posepile.util.videoproc as videoproc
from posepile import util
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler, AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example

BML_MOVI_ROOT = f'{DATA_ROOT}/bml_movi'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-affine-weights', action=spu.argparse.BoolAction)
    parser.add_argument('--calibrate-cameras', action=spu.argparse.BoolAction)
    parser.add_argument('--generate-dataset', action=spu.argparse.BoolAction)
    parser.add_argument('--stage', type=int, default=0)
    spu.initialize(parser)
    if FLAGS.generate_affine_weights:
        generate_affine_weights()
    if FLAGS.calibrate_cameras:
        calibrate_cameras()

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    video_filepaths = sorted(
        glob.glob(f'{BML_MOVI_ROOT}/*.mp4') + glob.glob(f'{BML_MOVI_ROOT}/*.avi'))
    video_path = video_filepaths[i_task]
    video_filename = osp.basename(video_path)
    logger.info(video_filename)
    out_path = (f'{DATA_ROOT}/bml_movi_downscaled/stage1/'
                f'{osp.splitext(video_filename)[0]}.pkl')
    if osp.exists(out_path):
        return

    video_relpath = osp.relpath(video_path, BML_MOVI_ROOT)
    detection_relpath = spu.replace_extension(video_relpath, '.pkl')
    detection_path = f'{BML_MOVI_ROOT}/detections/{detection_relpath}'
    all_detections = spu.load_pickle(detection_path)

    m = re.match(r'(?P<prefix>F|S[12])_(?P<camera_name>(CP|PG)[12])_'
                 r'Subject_(?P<i_subj>\d+)(_L)?.(mp4|avi)', video_filename)
    prefix = m['prefix']
    camera_name = m['camera_name']
    i_subj = m['i_subj']
    offset, cameras = load_offset_and_camera(prefix, camera_name, i_subj)

    gt_poses_all_markers = load_interpolated_gt_poses(
        prefix, i_subj, fps_cam=videoproc.get_fps(video_path), offset=offset,
        only_virtual_markers=False)

    examples = []
    pose_sampler = AdaptivePoseSampler2(
        100, assume_nan_unchanged=True, check_validity=True, buffer_size=500)

    with imageio.get_reader(video_path) as frames:
        for i_frame, (frame, world_all_markers, camera, detections) in enumerate(
                zip(frames, gt_poses_all_markers, cameras, all_detections)):
            if (camera is None or
                    camera_roll_degrees(camera) > 7 or
                    pose_sampler.should_skip(world_all_markers)):
                continue

            imcoords = camera.world_to_image(world_all_markers)
            gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.05)
            if detections.size > 0:
                i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in detections])
                box = detections[i_det][:4]
            else:
                box = gt_box

            new_image_replath = f'bml_movi_downscaled/{video_filename}/{i_frame:06d}.jpg'
            ex = ds3d.Pose3DExample(frame, world_all_markers, bbox=box, camera=camera)
            examples.append(make_efficient_example(ex, new_image_replath))

    examples.sort(key=lambda ex: ex.image_path)
    spu.dump_pickle(examples, out_path)


@spu.picklecache('bml_movi.pkl', min_time="2022-01-12T02:05:07")
def make_dataset():
    # virtual markers:
    # joint_names = (
    #    'head,mhip,pelv,thor,lank,lelb,lhip,lhan,lkne,lsho,lwri,lfoo,rank,relb,rhip,rhan,rkne,'
    #    'rsho,rwri,rfoo')
    # edges =
    names = (
        'backneck,upperback,clavicle,sternum,umbilicus,lfronthead,lbackhead,lback,lshom,'
        'lupperarm,lelbm,lforearm,lwrithumbside,lwripinkieside,lfin,lasis,lpsis,lfrontthigh,'
        'lthigh,lknem,lankm,lhee,lfifthmetatarsal,ltoe,lcheek,lbreast,lelbinner,lwaist,lthumb,'
        'lfrontinnerthigh,linnerknee,lshin,lfirstmetatarsal,lfourthtoe,lscapula,lbum,rfronthead,'
        'rbackhead,rback,rshom,rupperarm,relbm,rforearm,rwrithumbside,rwripinkieside,rfin,'
        'rasis,rpsis,rfrontthigh,rthigh,rknem,rankm,rhee,rfifthmetatarsal,rtoe,rcheek,rbreast,'
        'relbinner,rwaist,rthumb,rfrontinnerthigh,rinnerknee,rshin,rfirstmetatarsal,rfourthtoe,'
        'rscapula,rbum,head,mhip,pelv,thor,lank,lelb,lhip,lhan,lkne,lsho,lwri,lfoo,rank,relb,'
        'rhip,rhan,rkne,rsho,rwri,rfoo')
    edges = 'head-thor-pelv-mhip,thor-rsho-relb-rwri-rhan,mhip-rhip-rkne-rank-rfoo'

    joint_info = JointInfo(names, edges)
    example_paths = sorted(glob.glob(f'{DATA_ROOT}/bml_movi_downscaled/stage1/*.pkl'))
    examples = [ex for p in example_paths for ex in spu.load_pickle(p)]
    ds = ds3d.Pose3DDataset(joint_info, examples)
    ds3d.add_masks(ds, f'{DATA_ROOT}/bml_movi_downscaled/masks', 2)
    return ds


def camera_roll_degrees(cam):
    forward = cam.R[2]
    right_expected = np.cross(forward, cam.world_up)
    right_expected /= np.linalg.norm(right_expected)
    right_actual = cam.R[0]
    cos = np.clip(right_expected.T @ right_actual, -1, 1)
    roll_degrees = np.rad2deg(np.arccos(cos))
    return roll_degrees


def calibrate_cameras():
    # Intrinsics is fixed to a value obtained in preliminary experimentation.
    # When calibrating such that the intrinsic matrix wasn't fixed,
    # most videos calibrated to this focal length. For simplicity, we assume the principal point
    # to be the center.
    intr = np.array([[1508.7021, 0, 1920 / 2], [0, 1508.7021, 1080 / 2], [0, 0, 1]])

    for prefix, camera_name in itertools.product(('F', 'S1', 'S2'), ('CP1', 'CP2')):
        out_path = f'{BML_MOVI_ROOT}/Calib/{prefix}_{camera_name}.pkl'
        if osp.exists(out_path):
            continue

        subjects = glob.glob(f'{BML_MOVI_ROOT}/{prefix}_{camera_name}_Subject_*.mp4')
        subjects = sorted([s.split('_')[-1].split('.')[0] for s in subjects])
        cam_and_offsets = {}
        with spu.ThrottledPool() as pool:
            for s in spu.progressbar(subjects, desc=f'{prefix}_{camera_name}'):
                pool.apply_async(
                    calibrate_cam_and_offset, (prefix, s, camera_name, intr),
                    callback=spu.itemsetter(cam_and_offsets, s))
        spu.dump_pickle(cam_and_offsets, out_path)


def generate_affine_weights():
    gt2d_v, pred2d_v = load_matching_gt_and_pred2d()
    pose_sampler = AdaptivePoseSampler(50)
    indices = [i for i, pose in enumerate(gt2d_v) if not pose_sampler.should_skip(pose)]
    W = solve_for_affine_weights(pred2d_v[indices], gt2d_v[indices])
    np.save(f'{BML_MOVI_ROOT}/latent_to_bmlmovi.npy', W)


def solve_for_affine_weights(ins, outs, stride=5, lambdval=1e-1):
    means = np.nanmean(outs, axis=1, keepdims=True)
    stdevs = np.nanstd(outs[:, :, 1:2], axis=1, keepdims=True)
    ins = (ins - means) / stdevs
    outs = (outs - means) / stdevs
    ins = einops.rearrange(ins, 'b j c -> (b c) j')
    outs = einops.rearrange(outs, 'b j c -> (b c) j')
    return solve_affine_lasso(ins[::stride], outs[::stride], lambdval)


def load_matching_gt_and_pred2d():
    matfiles = glob.glob(f'{BML_MOVI_ROOT}/F_Subjects_*/F_v3d_Subject_*.mat')
    all_to_latent = np.load(f'{DATA_ROOT}/skeleton_conversion/all_to_latent_32_singlestage.npy')
    gt2ds = []
    pred2ds = []
    for i_cam in (1, 2):
        for matfile in spu.progressbar(matfiles):
            name = osp.basename(matfile)
            i_subj = name.split('_')[-1].split('.')[0]
            try:
                pred3d, pred2d = load_predictions('F', i_subj, f'PG{i_cam}')
            except FileNotFoundError:
                continue

            pred2d = [[convert_pose(pose, all_to_latent) for pose in poses] for poses in pred2d]
            camera = load_pg_camera(f'PG{i_cam}')
            gt3d = load_gt_poses('F', i_subj)[::4]
            if gt3d.shape[0] != len(pred2d):
                continue
            is_pred_valid = np.array([len(p) > 0 for p in pred2d])
            is_gt_valid = np.array([np.all(geom3d.are_joints_valid(a)) for a in gt3d])
            is_valid = np.logical_and(is_gt_valid, is_pred_valid)
            if not np.any(is_valid):
                continue
            pred2d_v = np.array([pred2d[i][0] for i, v in enumerate(is_valid) if v])
            gt3d_v = gt3d[is_valid]
            gt2d_v = np.array([camera.world_to_image(w) for w in gt3d_v])
            gt2ds.append(gt2d_v)
            pred2ds.append(pred2d_v)

    return np.concatenate(gt2ds, axis=0), np.concatenate(pred2ds, axis=0)


def calibrate_extrinsics_simple(image_coords2d, world_coords3d, intrinsic_matrix):
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
             cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
             cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH)
    is_valid = geom3d.are_joints_valid(world_coords3d)
    image_coords2d = image_coords2d[is_valid]
    world_coords3d = world_coords3d[is_valid]
    coords2d = image_coords2d[np.newaxis].astype(np.float32)
    coords3d = world_coords3d[np.newaxis].astype(np.float32)

    reproj_error, intrinsic_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        coords3d, coords2d, cameraMatrix=intrinsic_matrix, imageSize=(1920, 1080),
        distCoeffs=None, flags=flags)

    rot_matrix = cv2.Rodrigues(rvecs[0])[0]
    t = tvecs[0]
    extrinsic_matrix = np.concatenate([rot_matrix, t], axis=1)
    return cameralib.Camera(intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)


def calibrate_extrinsics(
        points2d, points3d, intrinsic_matrix=np.eye(3), inlier_thresh=15, rng=None, try_again=True):
    def find_inliers(cam):
        reprojected = cam.world_to_image(points3d)
        errors = np.linalg.norm(reprojected - points2d, axis=-1)
        auc = np.mean(np.maximum(0, 1 - errors / inlier_thresh))
        return errors < inlier_thresh, auc

    i_points = np.array(list(range(len(points2d))))
    n_iter = 100
    best_inliers = []
    best_auc = 0
    for i in range(n_iter):
        selected_points = list(rng.choice(i_points, size=6, replace=False))
        cam = calibrate_extrinsics_simple(
            points2d[selected_points], points3d[selected_points], intrinsic_matrix)
        is_inlier, auc = find_inliers(cam)
        inlier_ratio = np.mean(is_inlier)
        if auc > best_auc:
            best_inliers = np.nonzero(is_inlier)[0]
            best_auc = auc
        if inlier_ratio == 1:
            break

    # print(best_auc, 'auc')
    if len(best_inliers) >= 6:
        cam = calibrate_extrinsics_simple(
            points2d[best_inliers], points3d[best_inliers], intrinsic_matrix)
        is_inlier, auc = find_inliers(cam)
        # print(auc, 'auc final')
        return cam
    elif try_again:
        print('trying again')
        return calibrate_extrinsics(
            points2d, points3d, intrinsic_matrix, inlier_thresh=30, try_again=False, rng=rng)
    else:
        print('Not enough inliers')
        return None


def load_pg_camera(name):
    extr = np.load(f'{BML_MOVI_ROOT}/Calib/Extrinsics_{name}.npz')
    rotmat = extr['rotationMatrix'].T
    transvec = extr['translationVector']
    optical_center = -rotmat.T @ transvec
    intr_data = np.load(f'{BML_MOVI_ROOT}/Calib/cameraParams_{name}.npz')
    intr_mat = intr_data['IntrinsicMatrix'].T
    distortion_coeffs = np.zeros(5)
    distortion_coeffs[:2] = intr_data['RadialDistortion']
    return cameralib.Camera(
        intrinsic_matrix=intr_mat, rot_world_to_cam=rotmat,
        optical_center=optical_center, distortion_coeffs=distortion_coeffs)


def load_offset_and_camera(prefix, camera_name, i_subj):
    if camera_name.startswith('PG'):
        camera = load_pg_camera(camera_name)
        return 0, itertools.repeat(camera)

    info = spu.load_pickle(f'{BML_MOVI_ROOT}/Calib/{prefix}_{camera_name}.pkl')[i_subj]
    cameras = info['camera']
    if not isinstance(cameras, dict):
        camera = cameras
        cameras = itertools.repeat(camera)
    else:
        cameras = [cameras.get(i_frame) for i_frame in range(max(cameras.keys()) + 1)]
    return info['time_offset'], cameras


def load_interpolated_gt_poses(prefix, i_subj, fps_cam, offset, only_virtual_markers=True):
    coords_gt = load_gt_poses(prefix, i_subj, only_virtual_markers)
    n_joints = 20 if only_virtual_markers else 87
    gt_poses = []
    for i_frame in itertools.count():
        gt_index = (i_frame / fps_cam - offset) * 120
        if gt_index < 0:
            gt_poses.append(np.full((n_joints, 3), np.nan))
            continue

        if gt_index >= len(coords_gt) - 1:
            break
        gt_pose = interpolate(coords_gt, gt_index)
        gt_poses.append(gt_pose)

    return np.array(gt_poses, np.float32)


def load_gt_poses(prefix, i_subj, only_virtual_markers=True):
    prefix0 = prefix[0]
    if prefix0 == 'S':
        relpath = f'S/S_v3d_Subject_{i_subj}.mat'
    else:
        subdir = 'F_Subjects_1_45' if int(i_subj) <= 45 else 'F_Subjects_46_90'
        relpath = f'{subdir}/F_v3d_Subject_{i_subj}.mat'
    path = f'{BML_MOVI_ROOT}/{relpath}'
    arr = util.load_mat(path)
    move = arr[f'Subject_{i_subj}_{prefix0}'].move
    if prefix == 'F':
        coords = move.virtualMarkerLocation if only_virtual_markers else move.markerLocation
    else:
        num = int(prefix[1]) - 1

        coords_subset = (
            move[num].virtualMarkerLocation if only_virtual_markers else move[num].markerLocation)
        indices_out = (
            [0, 4, 10, 12, 18] if only_virtual_markers
            else [5, 6, 12, 13, 20, 36, 37, 43, 44, 51, 67, 71, 77, 79, 85])
        indices_in = (
            range(len(indices_out))
            if only_virtual_markers else [i for i in range(17) if i not in [5, 11]])

        n_joints = 20 if only_virtual_markers else 87
        coords = np.full(
            (coords_subset.shape[0], n_joints, 3), fill_value=np.nan)
        for i_out, i_in in zip(indices_out, indices_in):
            coords[:, i_out] = coords_subset[:, i_in]

    coords[coords < 1e-3] = np.nan
    return coords


def load_predictions(prefix, i_subj, cam_name):
    suf = '_L' if cam_name.startswith('PG') else ''
    pred_path = f'{BML_MOVI_ROOT}/pred/{prefix}_{cam_name}_Subject_{i_subj}{suf}.pkl'
    preds = spu.load_pickle(pred_path)
    return preds['poses3d'], preds['poses2d']


def solve_affine_lasso(X, Y, lambdval):
    import cvxpy as cp
    lambdval = 32 / X.shape[1] * lambdval

    def objective_fn(X, Y, beta, lambd):
        return 1 / (2 * len(X)) * cp.norm2(X @ beta - Y) ** 2 + lambd * cp.norm1(beta)

    def constraint(beta):
        return cp.sum(beta) == 1

    beta = cp.Variable(X.shape[1])
    lambd = cp.Parameter(nonneg=True)
    betas = []
    lambd.value = lambdval

    X_valid = np.all(~np.isnan(X), axis=-1)
    Y_valid = ~np.isnan(Y)

    for i_target in spu.progressbar(range(Y.shape[1])):
        target = Y[:, i_target]
        valid = np.logical_and(Y_valid[:, i_target], X_valid)
        problem = cp.Problem(
            cp.Minimize(objective_fn(X[valid], target[valid], beta, lambd)), [constraint(beta)])
        problem.solve(solver='SCS', max_iters=200000, eps=1e-7)
        betas.append(beta.value)
    W = np.array(betas).T
    return W


def convert_pose(coords, weights):
    return np.einsum('...jc,jJ->...Jc', coords, weights)


def calibrate_with_offset(offset, coords_gt, pred3d, pred2d, fps_cam, intr, rng):
    i_frames, gt_poses, pred_poses3d, pred_poses2d = get_interpolated_pose_pairs(
        offset, coords_gt, pred3d, pred2d, fps_cam)
    gt_poses = rigid_align_filled(pred_poses3d, gt_poses)
    return calibrate_extrinsics(
        np.array(pred_poses2d.reshape(-1, 2)), np.array(gt_poses.reshape(-1, 3)), intr, rng=rng)


def interpolate(arr, i):
    floor = int(np.floor(i))
    if floor == i:
        return arr[floor]

    weight2 = i - floor
    return arr[floor] * (1 - weight2) + arr[floor + 1] * weight2


def get_interpolated_pose_pairs(offset, coords_gt, pred3d, pred2d, fps_cam):
    i_frames = []
    gt_poses = []
    pred_poses3d = []
    pred_poses2d = []

    for i_frame, (pred_frame3d, pred_frame2d) in enumerate(zip(pred3d, pred2d)):
        gt_index = (i_frame / fps_cam - offset) * 120
        if gt_index < 0 or len(pred_frame3d) == 0:
            continue
        if gt_index >= len(coords_gt) - 1:
            break

        gt_pose = interpolate(coords_gt, gt_index)
        if np.count_nonzero(geom3d.are_joints_valid(gt_pose)) > 3:
            i_frames.append(i_frame)
            gt_poses.append(gt_pose)
            pred_poses3d.append(pred_frame3d[0])
            pred_poses2d.append(pred_frame2d[0])

    return (i_frames,
            np.array(gt_poses, np.float32),
            np.array(pred_poses3d, np.float32),
            np.array(pred_poses2d, np.float32))


def procrustes_err_for_offset(offset, coords_gt, pred3d, pred2d, fps_cam):
    i_frames, gt_poses, pred_poses3d, pred_poses2d = get_interpolated_pose_pairs(
        offset, coords_gt, pred3d, pred2d, fps_cam)

    all3d_pred, all3d_gt = rigid_align_get_valid_points(pred_poses3d, gt_poses)
    return np.mean(np.linalg.norm(all3d_gt - all3d_pred, axis=-1))


def to_dense(poses3d, n_joints):
    arr = np.full(shape=[len(poses3d), n_joints, 3], dtype=np.float32, fill_value=np.nan)
    for i, poses_of_frame in enumerate(poses3d):
        if len(poses_of_frame) > 0:
            arr[i] = poses_of_frame[0]
    return arr


def calibrate_cam_and_offset(prefix, i_subj, camera_name, intr):
    # print(prefix, i_subj, camera_name)
    coords_gt = load_gt_poses(prefix, i_subj)
    pred3d, pred2d = load_predictions(prefix, i_subj, camera_name)
    video_path = f'{BML_MOVI_ROOT}/F_{camera_name}_Subject_{i_subj}.mp4'
    fps = videoproc.get_fps(video_path)
    all_to_latent = np.load(f'{DATA_ROOT}/skeleton_conversion/all_to_latent_32_singlestage.npy')
    W = np.load(f'{BML_MOVI_ROOT}/latent_to_bmlmovi.npy')
    all_to_bmlmovi = all_to_latent @ W
    pred3d = [[convert_pose(pose, all_to_bmlmovi) for pose in poses] for poses in pred3d]
    pred2d = [[convert_pose(pose, all_to_bmlmovi) for pose in poses] for poses in pred2d]

    coords_pred = to_dense(pred3d, 20)
    t_move_pred = find_first_movement(coords_pred, thresh=300) / fps
    t_move_gt = find_first_movement(coords_gt, thresh=300) / 120
    # print(t_move_pred, t_move_gt)
    approx_offset = t_move_pred - t_move_gt

    def objective_fn(x):
        err = procrustes_err_for_offset(x, coords_gt, pred3d, pred2d, fps)
        # print(x, err)
        return err

    def optimize(bounds):
        return scipy.optimize.minimize_scalar(
            objective_fn, bounds=bounds, method='bounded', options=dict(xatol=1 / 240))

    res = optimize((approx_offset - 0.5, approx_offset + 0.5))
    if res.fun > 100:
        # print('Simple alignment did not work')
        results = [optimize(bounds) for bounds in ((-15, 0), (0, 15), (15, 30), (30, 45))]
        i_best = np.argmin([r.fun for r in results])
        res = results[i_best]
        if res.fun > 100:
            # print('Doing exhaustive check')
            offs = np.linspace(-20, 40, 300)
            errs = [objective_fn(off) for off in offs]
            approx_offset = offs[np.argmin(errs)]
            res = optimize((approx_offset - 0.5, approx_offset + 0.5))
            if res.fun > 100:
                raise RuntimeError('Error too high!')

    optimal_offset = res.x

    rng = np.random.Generator(np.random.PCG64(1234))
    if camera_name == 'CP1':
        cam = calibrate_with_offset(optimal_offset, coords_gt, pred3d, pred2d, fps, intr, rng=rng)
    else:
        cam = calibrate_each_with_offset(
            optimal_offset, coords_gt, pred3d, pred2d, fps, intr, rng=rng)
    return dict(time_offset=optimal_offset, camera=cam)


def find_first_movement(poses, thresh=100):
    pose_sampler = AdaptivePoseSampler(thresh, assume_nan_unchanged=True)
    return next((i for i, p in enumerate(poses)
                 if not pose_sampler.should_skip(p) and i > 0), 0)


def calibrate_each_with_offset(offset, coords_gt, pred3d, pred2d, fps_cam, intr, rng):
    i_frames, gt_poses, pred_poses3d, pred_poses2d = get_interpolated_pose_pairs(
        offset, coords_gt, pred3d, pred2d, fps_cam)
    gt_poses = rigid_align_filled(pred_poses3d, gt_poses)
    return {
        i_frame: calibrate_extrinsics(np.array(pred_pose2d), np.array(gt_pose), intr, rng=rng)
        for i_frame, gt_pose, pred_pose2d in zip(i_frames, gt_poses, pred_poses2d)}


def rigid_align_filled(pred_coords, gt_coords):
    aligned, is_valids = rigid_alignment.rigid_align(pred_coords, gt_coords)
    gt_coords = gt_coords.copy()
    gt_coords[~is_valids] = aligned[~is_valids]
    return gt_coords


def rigid_align_get_valid_points(pred_coords, gt_coords):
    aligned, is_valids = rigid_alignment.rigid_align(pred_coords, gt_coords)
    return aligned[is_valids].reshape(-1, 3), gt_coords[is_valids].reshape(-1, 3)


if __name__ == '__main__':
    main()
