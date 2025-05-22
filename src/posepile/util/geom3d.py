import numpy as np

import posepile.joint_info


def are_joints_valid(coords):
    return np.logical_not(np.any(np.isnan(coords), axis=-1))


def scale_align(poses):
    mean_scale = np.sqrt(np.mean(np.square(poses), axis=(-3, -2, -1), keepdims=True))
    scales = np.sqrt(np.mean(np.square(poses), axis=(-2, -1), keepdims=True))
    return poses / scales * mean_scale


def geometric_median(X, eps=5):
    from scipy.spatial.distance import cdist, euclidean
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def convert_pose(coords, weights):
    return np.einsum('...jc,jJ->...Jc', coords, weights)


def are_bones_plausible(
        poses, reference_bone_lengths, joint_info=None, relsmall_thresh=0.1, relbig_thresh=3,
        absbig_thresh=150, joints2bones_matrix=None):
    if joints2bones_matrix is None:
        joints2bones_matrix = posepile.joint_info.get_joint2bone_mat(joint_info)

    joints2bones_abs = np.abs(joints2bones_matrix)

    is_joint_valid = are_joints_valid(poses).astype(np.float32)
    is_bone_valid = joints2bones_abs @ is_joint_valid[..., np.newaxis] == 2
    is_bone_valid = np.squeeze(is_bone_valid, -1)
    bones = joints2bones_matrix @ np.nan_to_num(poses)
    bone_lengths = np.linalg.norm(bones, axis=-1)
    bone_lengths[~is_bone_valid] = np.nan
    denominator = 1 / (reference_bone_lengths + 1e-8)

    bone_length_relative = bone_lengths * denominator
    bone_length_diff = np.abs(bone_lengths - reference_bone_lengths)

    with np.errstate(invalid='ignore'):
        relsmall = bone_length_relative < relsmall_thresh
        relbig = bone_length_relative > relbig_thresh
        absdiffbig = bone_length_diff > absbig_thresh

    is_bone_implausible = np.logical_and(np.logical_or(relbig, relsmall), absdiffbig)
    return np.logical_not(is_bone_implausible)


def get_scale(poses, keepdims=False):
    return np.sqrt(np.mean(np.square(poses), axis=(-2, -1), keepdims=keepdims))


def confidence(stdev, half_point=50):
    x = 0.8325 / half_point * stdev
    return np.exp(-x ** 2)


def scale_align_to_true(pred, true):
    return pred / get_scale(pred, keepdims=True) * get_scale(true, keepdims=True)


def point_stdev(poses, item_axis=-3, coord_axis=-1):
    mean = np.mean(poses, axis=item_axis, keepdims=True)
    n_items = poses.shape[item_axis]
    corr_factor = n_items / (n_items - 1) if n_items > 1 else 1
    result = np.sqrt(corr_factor * np.mean(
        np.square(poses - mean), axis=(item_axis, coord_axis), keepdims=True))
    return np.squeeze(result, (item_axis, coord_axis))


def relu(x):
    return np.maximum(0, x)


def auc(x, t1, t2):
    return relu(np.float32(1) - relu(x - t1) / (t2 - t1))


def unit_vector(vectors, axis=-1):
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / norm
