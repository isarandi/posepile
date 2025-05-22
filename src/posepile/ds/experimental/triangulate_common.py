import collections
import functools

import boxlib
import cameralib
import cv2
import numba
import numpy as np
import rlemasklib
import scipy.optimize
import simplepyutils as spu

import src.posepile.util.drawing as drawing
import posepile.util.maskproc as maskproc
import posepile.util.rigid_alignment as rigid_alignment
from posepile.util import geom3d


def triangulate_sequence(
        i_ref_cam, out_path, poses3d_per_cam, gt3d_per_cam, boxes_per_cam, cameras,
        dynamic_time_warping=False, already_calibrated=False, n_ignore_end_warp=500):
    resampler_fn = (
        functools.partial(warp3, cameras=cameras, n_ignore_end=n_ignore_end_warp)
        if dynamic_time_warping
        else resample3_by_len)

    resamps, indices = resampler_fn(poses3d_per_cam, i_ref=i_ref_cam)

    if already_calibrated:
        cameras_calib = cameras
    else:
        cameras_calib = [
            cameras[0],
            calibrate_from_poses(resamps[0], resamps[1], cameras[1]),
            calibrate_from_poses(resamps[0], resamps[2], cameras[2])]

    imposes = project(resamps)
    triangs = np.array([
        triangulate_multiview_per_point([cameras_calib[c] for c in i_cams], imposes[i_cams])
        for i_cams in [[0, 1, 2], [0, 1], [0, 2]]])

    reproj_errors = np.stack([
        calc_reproj_error(triangs, resamps[i_cam], cameras_calib[i_cam])
        for i_cam in range(3)], axis=1)

    median_err_over_cameras = np.nanmedian(reproj_errors, axis=1)
    choices = infargmin(median_err_over_cameras, axis=0)
    triangs_combined = np.choose(choices[..., np.newaxis], triangs)

    triple_triang_is_good_in_all = np.all(reproj_errors[0] < 60, axis=0)
    triangs_combined[triple_triang_is_good_in_all] = triangs[0][
        triple_triang_is_good_in_all]

    min_err = infmin(median_err_over_cameras, axis=0)
    resamps_gt = [resample(gt3d, ind) for gt3d, ind in zip(gt3d_per_cam, indices)]
    content = dict(
        camera=cameras_calib[i_ref_cam],
        poses3d=triangs_combined,
        boxes=boxes_per_cam[i_ref_cam],
        original_kinect_poses3d=resamps_gt[i_ref_cam],
        errors=min_err,
        frame_indices=indices)

    spu.dump_pickle(content, out_path)


def pred_to_masked_avg_poses_assoc(
        pred3d, gt3d, boxes_det, masks, camera, n_people, joint_info, joint_info_gt,
        stdev_thresh=40):
    _, n_aug, n_joints, n_coord = pred3d[0].shape
    poses_out = np.full(
        [len(pred3d), n_people, n_aug, n_joints, 3], fill_value=np.nan, dtype=np.float32)
    dets_out = np.full([len(pred3d), n_people, 5], fill_value=np.nan, dtype=np.float32)
    gt_out = np.full(
        [len(pred3d), n_people, joint_info_gt.n_joints, 3], fill_value=np.nan, dtype=np.float32)

    frame_shape = [1080, 1920]

    for i_frame, (poses_in_frame, gt_in_frame, dets_in_frame, masks_in_frame) in enumerate(
            zip(pred3d, gt3d, boxes_det, masks)):
        poses_in_frame2d = camera.camera_to_image(np.mean(poses_in_frame, axis=-3))
        gt_in_frame2d = camera.camera_to_image(gt_in_frame)

        indices = associate_poses_to_masks(
            poses_in_frame2d, frame_shape, masks_in_frame, joint_info)
        for i_dst, i_src in enumerate(indices):
            if i_src is not None:
                poses_out[i_frame, i_dst] = poses_in_frame[i_src]
                dets_out[i_frame, i_dst] = dets_in_frame[i_src]

        indices_gt = associate_poses_to_masks(
            gt_in_frame2d, frame_shape, masks_in_frame, joint_info_gt)
        for i_dst, i_src in enumerate(indices_gt):
            if i_src is not None:
                gt_out[i_frame, i_dst] = gt_in_frame[i_src]

    return mask_and_average(poses_out, stdev_thresh), dets_out, gt_out


def associate_poses_to_masks(poses2d_pred, frame_shape, masks, joint_info3d):
    mask_shape = masks[0]['size']
    mask_size = np.array([mask_shape[1], mask_shape[0]], np.float32)
    frame_size = np.array([frame_shape[1], frame_shape[0]], np.float32)
    poses2d_pred = poses2d_pred * mask_size / frame_size
    pose_masks = [rlemasklib.encode(pose_to_mask(p, mask_shape, joint_info3d, 8))
                  for p in poses2d_pred]
    iou_matrix = np.array([[rlemasklib.iou([m1, m2]) for m2 in pose_masks] for m1 in masks])
    true_indices, pred_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
    n_true_poses = len(masks)

    indices = [None] * n_true_poses
    for ti, pi in zip(true_indices, pred_indices):
        indices[ti] = pi
    return indices


def pose_to_mask(pose2d, imshape, joint_info, thickness, thresh=0.2):
    result = np.zeros(imshape[:2], dtype=np.uint8)
    if pose2d.shape[1] == 3:
        is_valid = pose2d[:, 2] > thresh
    else:
        is_valid = geom3d.are_joints_valid(pose2d)

    for i_joint1, i_joint2 in joint_info.stick_figure_edges:
        if is_valid[i_joint1] and is_valid[i_joint2]:
            drawing.line(
                result, pose2d[i_joint1, :2], pose2d[i_joint2, :2], color=(1, 1, 1),
                thickness=thickness)

    j = joint_info.ids
    torso_joints = [j.lhip, j.rhip, j.rsho, j.lsho]
    if np.all(is_valid[torso_joints]):
        drawing.fill_polygon(result, pose2d[torso_joints, :2], (1, 1, 1))
    return result


def mask_bad(triang, err, thresh=35):
    triang = triang.copy()
    triang[err > thresh] = np.nan
    return triang


def calibrate_from_points(
        points_world, points_cam, camera, cam_world=None, return_inlier_mask=False):
    points2d_world = project(points_world)
    points2d_cam = project(points_cam)
    eye = np.eye(3, dtype=np.float32)

    essential_matrix, inlier_mask = cv2.findEssentialMat(
        points2d_world, points2d_cam, eye, method=cv2.LMEDS,
        prob=1 - 1e-15, maxIters=2147483647)  # , threshold=30 / camera.intrinsic_matrix[0, 0])
    # prob=1 - 1e-11, maxIters=2147483647)

    # essential_matrix, inlier_mask = cv2.findEssentialMat(
    #     points2d_world, points2d_cam, eye, method=cv2.RANSAC,
    #     prob=1 - 1e-11, maxIters=2147483647, threshold=100 / camera.intrinsic_matrix[0, 0])
    # # prob=1 - 1e-11, maxIters=2147483647)

    inlier_mask = inlier_mask.squeeze(1) == 1
    print(np.mean(inlier_mask))

    _, R, t, _, triangulated = cv2.recoverPose(
        essential_matrix, points2d_world[inlier_mask],
        points2d_cam[inlier_mask], eye, distanceThresh=10000)
    triangulated = (triangulated[:3] / triangulated[3:]).T

    scale_world = geom3d.get_scale(points_world[inlier_mask])
    scale_triang = geom3d.get_scale(triangulated)
    scale_factor = scale_world / scale_triang

    relative_extr = np.block([[R, t * scale_factor], [0, 0, 0, 1]]).astype(np.float32)
    extrinsics = relative_extr @ (cam_world.get_extrinsic_matrix()
                                  if cam_world is not None else np.eye(4, dtype=np.float32))
    camera_new = cameralib.Camera(
        extrinsic_matrix=extrinsics, intrinsic_matrix=camera.intrinsic_matrix,
        distortion_coeffs=camera.distortion_coeffs, world_up=camera.world_up)

    if return_inlier_mask:
        return camera_new, inlier_mask
    return camera_new


def calibrate_from_poses(poses_world, poses_cam, cam, cam_world=None, return_error=False):
    points_world = poses_world.reshape(-1, 3)
    points_cam = poses_cam.reshape(-1, 3)
    is_valid = np.logical_and(
        geom3d.are_joints_valid(points_world),
        geom3d.are_joints_valid(points_cam))
    points_world_valid = points_world[is_valid]
    points_cam_valid = points_cam[is_valid]

    cam_new, inlier_mask = calibrate_from_points(
        points_world_valid, points_cam_valid, cam, cam_world=cam_world, return_inlier_mask=True)
    if cam_world is not None:
        aligned = cam_new.world_to_camera(cam_world.camera_to_world(points_world_valid))
    else:
        aligned = cam_new.world_to_camera(points_world_valid)
    err = np.mean(np.linalg.norm(project(aligned) - project(points_cam_valid), axis=-1)) * \
          cam.intrinsic_matrix[0, 0]
    print('Calib err:', err)
    inlier_err = np.mean(
        np.linalg.norm(
            project(aligned[inlier_mask]) - project(points_cam_valid[inlier_mask]), axis=-1)) * \
                 cam.intrinsic_matrix[0, 0]
    print('Inlier err:', inlier_err)
    if return_error:
        return cam_new, err, inlier_err
    return cam_new


def resample3(poses1, poses2, poses3, factor2=1, factor3=1):
    results = np.empty([3, *poses1.shape], np.float32)
    results[0] = poses1
    for i in range(len(poses1)):
        results[1, i] = interpolate(poses2, i * factor2)
        results[2, i] = interpolate(poses3, i * factor3)

    factors = np.array([1, factor2, factor3], np.float32)[:, np.newaxis]
    indices = np.arange(len(poses1), dtype=np.float32) * factors
    return results, indices


def resample3_by_len(poses_per_cam, i_ref=0):
    reflen = len(poses_per_cam[i_ref])
    nonrefs = [poses_per_cam[i] for i in range(3) if i != i_ref]
    i_nonrefs = [i for i in range(3) if i != i_ref]
    outs, indices = resample3(
        poses_per_cam[i_ref], nonrefs[0], nonrefs[1],
        (len(nonrefs[0]) - 1) / (reflen - 1),
        (len(nonrefs[1]) - 1) / (reflen - 1))
    order = np.argsort([i_ref, *i_nonrefs])
    return outs[order], indices[order]


def warp3(poses_per_cam, i_ref=0, cameras=None, n_ignore_end=500):
    reflen = len(poses_per_cam[i_ref])
    nonrefs = [poses_per_cam[i] for i in range(3) if i != i_ref]
    i_nonrefs = [i for i in range(3) if i != i_ref]
    outs, indices = zip(
        (poses_per_cam[i_ref], np.arange(reflen)),
        warp_time(poses_per_cam[i_ref], nonrefs[0], cameras[i_nonrefs[0]], n_ignore_end),
        warp_time(poses_per_cam[i_ref], nonrefs[1], cameras[i_nonrefs[1]], n_ignore_end))
    order = np.argsort([i_ref, *i_nonrefs])
    return np.array(outs)[order], np.array(indices)[order]


def calc_reproj_error(pose_world, pose_cam, cam):
    reprojected = project(cam.world_to_camera(pose_world))
    projected = project(pose_cam)
    z = pose_cam[..., 2]
    return np.linalg.norm(reprojected - projected, axis=-1) * z


def infmin(x, axis=None):
    if x.size == 0:
        return np.inf
    return np.min(np.nan_to_num(x, np.inf), axis=axis)


def infargmin(x, axis=None):
    return np.argmin(np.nan_to_num(x, np.inf), axis=axis)


def get_scale_factor(triang_poses, gt_poses):
    is_valid = np.logical_and(
        geom3d.are_joints_valid(triang_poses),
        geom3d.are_joints_valid(gt_poses))

    scale = np.array([
        rigid_alignment.procrustes(gt_pose[v], triang_pose[v], scaling=True)[2]['scale']
        for gt_pose, triang_pose, v in zip(gt_poses, triang_poses, is_valid)])
    return scale


def triangulate_multiview_per_point(cameras, pointsets, weights=None):
    pointsets = np.array(pointsets)
    weights = np.ones_like(pointsets[..., 0]) if weights is None else np.array(weights)

    if pointsets.ndim > 3:
        pointsets_reshaped = np.reshape(pointsets, [pointsets.shape[0], -1, 2])
        weights_reshaped = np.reshape(weights, [weights.shape[0], -1])
        triangulated_reshaped = triangulate_multiview_per_point(
            cameras, pointsets_reshaped, weights_reshaped)
        return np.reshape(triangulated_reshaped, [*pointsets.shape[1:-1], 3])

    proj_matrices = np.array([c.get_extrinsic_matrix()[:3] for c in cameras])
    n_points = len(pointsets[0])
    triangulated = np.empty(shape=(n_points, 3), dtype=np.float32)
    for i_point in range(n_points):
        is_valid = geom3d.are_joints_valid(pointsets[:, i_point])
        if np.count_nonzero(is_valid) >= 2:
            triangulated[i_point] = triangulate_point(
                pointsets[is_valid, i_point], proj_matrices[is_valid], weights[is_valid, i_point])
        else:
            triangulated[i_point] = np.nan
    return triangulated


def triangulate_point(point2d_per_views, proj_matrices, weights):
    blocks = [w * (np.expand_dims(point, 1) @ pr[2:] - pr[:2])
              for point, pr, w in zip(point2d_per_views, proj_matrices, weights)]
    A = np.concatenate(blocks, axis=0)
    vh = np.linalg.svd(A, full_matrices=False)[2]
    return vh[..., 3, :3] / vh[..., 3, 3:]


def interpolate(arr, i):
    if np.isnan(i):
        return np.full(arr.shape[1:], dtype=arr.dtype, fill_value=np.nan)
    floor = int(np.floor(i))
    if floor >= len(arr):
        return arr[-1]
    if floor == i or floor + 1 >= len(arr):
        return arr[floor]

    weight2 = i - floor
    return arr[floor] * (1 - weight2) + arr[floor + 1] * weight2


def pred_to_masked_avg_poses(pred3d, gt3d, boxes_det, stdev_thresh=40):
    def get_bbox(pose):
        norm_imcoords = pose[..., :2] / pose[..., 2:]
        return boxlib.bb_of_points(norm_imcoords)

    _, n_aug, n_joints, n_coord = pred3d[0].shape
    poses_out = np.full([len(pred3d), n_aug, n_joints, 3], fill_value=np.nan, dtype=np.float32)
    dets_out = np.full([len(pred3d), 5], fill_value=np.nan, dtype=np.float32)
    for i_frame, (poses_in_frame, gt_in_frame, dets_in_frame) in enumerate(
            zip(pred3d, gt3d, boxes_det)):
        pred_boxes = [get_bbox(np.mean(p, axis=0)) for p in poses_in_frame]

        if np.all(np.isnan(gt_in_frame)):
            continue

        gt_boxes = [get_bbox(p) for p in [gt_in_frame]]
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue

        iou_matrix = np.array(
            [[boxlib.iou(pred_box, gt_box)
              for pred_box in pred_boxes]
             for gt_box in gt_boxes], np.float32)

        i_gts, i_preds = scipy.optimize.linear_sum_assignment(-iou_matrix)
        for i_gt, i_pred in zip(i_gts, i_preds):
            poses_out[i_frame] = poses_in_frame[i_pred]
            dets_out[i_frame] = dets_in_frame[i_pred]
            break

    return mask_and_average(poses_out, stdev_thresh), dets_out


def to_array_single_person(poses):
    _, n_aug, n_joints, n_coord = poses[0].shape
    arr = np.full([len(poses), n_aug, n_joints, n_coord], fill_value=np.nan, dtype=np.float32)
    for i, poses_in_frame in enumerate(poses):
        if len(poses_in_frame) > 0:
            arr[i] = poses_in_frame[0]
    return arr


def project(x):
    return x[..., :2] / x[..., 2:3]


def mask_and_average(poses3d, stdev_thresh=40, confidence_half_point=None):
    means3d = np.mean(poses3d, axis=-3)
    mean_depth = np.expand_dims(means3d, -3)[..., 2:3]
    projected = project(poses3d) * mean_depth
    stdevs = geom3d.point_stdev(projected, item_axis=-3, coord_axis=-1)
    mask = stdevs > stdev_thresh
    means3d[mask] = np.nan

    if confidence_half_point is None:
        return means3d

    conf = geom3d.confidence(stdevs, half_point=confidence_half_point)
    return np.concatenate([means3d, conf[..., np.newaxis]], axis=-1)


def warp_time(poses3d_ref, poses3d_calib, camera, n_ignore_end=500):
    import fastdtw  # pip install fastdtw

    def to_input(x):
        x = interpolate_nonfinite(x)
        x = x[..., :24, :]
        x = x / x[..., 2:]
        x = x.reshape(x.shape[0], -1)
        return x

    try:
        camera = calibrate_from_poses(poses3d_ref[:10], poses3d_calib[:10], camera)
    except:
        camera = calibrate_from_poses(poses3d_ref[250:260], poses3d_calib[250:260], camera)

    E = (camera.R @ cross_product_skew_matrix(camera.t)).T
    input_ref = to_input(poses3d_ref)
    input_calib = to_input(poses3d_calib)
    distance, path = fastdtw.fastdtw(
        input_ref, input_calib, dist=lambda a, b: epipolar_error(a, b, E))
    path = np.array(path)

    preds_calib_resamp, _ = resample_by_path(path, poses3d_ref, poses3d_calib)
    camera = calibrate_from_poses(
        poses3d_ref[:-n_ignore_end], preds_calib_resamp[:-n_ignore_end], camera)
    E = (camera.R @ cross_product_skew_matrix(camera.t)).T
    distance, path = fastdtw.fastdtw(
        input_ref, input_calib, dist=lambda a, b: epipolar_error(a, b, E))
    return resample_by_path(path, poses3d_ref, poses3d_calib)


def resample_by_path(path, values1, values2):
    indices = collections.defaultdict(list)
    for i, j in path:
        indices[i].append(j)
    result = np.full_like(values1, fill_value=np.nan)

    inds = np.full([values1.shape[0]], dtype=np.float32, fill_value=np.nan)
    for i1, i2s in indices.items():
        if i1 >= len(values1):
            print('weird', i1, len(values1))
            continue
        i2 = np.mean(i2s)
        result[i1] = interpolate(values2, i2)
        inds[i1] = i2

    return result, inds


def resample(arr, indices):
    result = np.full(shape=[indices.shape[0], *arr.shape[1:]], dtype=arr.dtype, fill_value=np.nan)
    for i_out, i_in in enumerate(indices):
        result[i_out] = interpolate(arr, i_in)

    return result


def fill_prev_if_false(arr, mask):
    result = arr.copy()
    prev_result_slice = None
    for mask_slice, result_slice in zip(mask, result):
        if prev_result_slice is not None:
            result_slice[~mask_slice] = prev_result_slice[~mask_slice]
        prev_result_slice = result_slice
    return result


def fill_prev_if_false_reversed(arr, mask):
    return np.flipud(fill_prev_if_false(np.flipud(arr), np.flipud(mask)))


def interpolate_nonfinite(coords):
    """Fills in any non-finite values (NaN or +/-infinity) by interpolating linearly between
    the neighboring finite values along axis 0. If there is no neighbor in one direction, then
    the other neighbor is replicated. If there is no finite value for some slice along axis 0,
    the values are filled with zeros."""

    is_valid = np.isfinite(coords)
    shape = [-1] + [1 for _ in coords.shape[1:]]
    length = coords.shape[0]
    own_index = np.arange(length).reshape(shape)

    own_index = np.broadcast_to(own_index, is_valid.shape)
    own_index = np.ascontiguousarray(own_index)
    own_index[0, ~is_valid[0]] = -1
    own_index[-1, ~is_valid[-1]] = length

    valid_index_before = fill_prev_if_false(own_index, is_valid)
    valid_index_after = fill_prev_if_false_reversed(own_index, is_valid)
    has_no_valid_before = valid_index_before < 0
    has_no_valid_after = valid_index_before >= length

    valid_index_before = np.maximum(valid_index_before, 0)
    valid_index_after = np.minimum(valid_index_after, length - 1)

    value_before = np.take_along_axis(coords, valid_index_before, axis=0)
    value_after = np.take_along_axis(coords, valid_index_after, axis=0)

    weight_before = ((valid_index_after - own_index).astype(np.float32) /
                     np.maximum(1, valid_index_after - valid_index_before))

    interpolated = weight_before * value_before + (1 - weight_before) * value_after

    interpolated[has_no_valid_before] = value_after[has_no_valid_before]
    interpolated[has_no_valid_after] = value_before[has_no_valid_after]
    interpolated[~np.isfinite(interpolated)] = 0
    return interpolated


@numba.njit
def epipolar_error(a, b, E):
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    n_joints = a.shape[0]
    epi_errors = np.empty(shape=(n_joints,), dtype=np.float32)
    for i in range(n_joints):
        epi_errors[i] = np.abs(a[i].T @ E @ b[i])
    return np.median(epi_errors)


def cross_product_skew_matrix(v):
    return np.cross(v, np.identity(v.shape[0]) * -1)
