import argparse
import os.path as osp

import numpy as np
import scipy.ndimage
import simplepyutils as spu
import tensorflow as tf
from simplepyutils import FLAGS

import posepile.util.procrustes_tf as procrustes_tf
import smpl.numpy
import smpl.tensorflow
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--testset-only', action=spu.argparse.BoolAction)
    parser.add_argument('--joint14', action=spu.argparse.BoolAction)
    parser.add_argument('--use-h36m-regressor', action=spu.argparse.BoolAction)
    parser.add_argument('--align-at-hips', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)

    true_pose_dict, true_betas_dict, true_trans_dict, true_gender_dict = get_all_gt_poses(
        FLAGS.testset_only)
    pred_pose_dict, pred_betas_dict, pred_trans_dict = get_all_pred_poses()
    all_true_pose = np.array(list(true_pose_dict.values()))
    all_true_betas = np.array(list(true_betas_dict.values()))
    all_true_trans = np.array(list(true_trans_dict.values()))
    all_true_gender = np.array(list(true_gender_dict.values()))

    relpaths = list(true_pose_dict.keys())
    all_pred_pose = np.array([pred_pose_dict[relpath] for relpath in relpaths])
    all_pred_betas = np.array([pred_betas_dict[relpath] for relpath in relpaths])
    all_pred_trans = np.array([pred_trans_dict[relpath] for relpath in relpaths])

    batch_size = 64
    ds = tf.data.Dataset.from_tensor_slices(
        dict(true_pose=all_true_pose, true_betas=all_true_betas, true_trans=all_true_trans,
             true_gender=all_true_gender, pred_pose=all_pred_pose, pred_betas=all_pred_betas,
             pred_trans=all_pred_trans)).batch(batch_size)

    all_per_joint_errs = []
    all_per_joint_errs_rootrel = []
    all_per_joint_errs_pa = []
    all_vertex_errs = []
    all_vertex_errs_rootrel = []
    all_vertex_errs_pa = []
    all_per_joint_ori_errs = []
    all_per_joint_ori_errs_pa = []

    for batch in spu.progressbar(ds, total=len(relpaths), step=batch_size):
        (per_joint_errs, per_joint_errs_rootrel, per_joint_errs_pa, vertex_errs,
         vertex_errs_rootrel, vertex_errs_pa, per_joint_ori_errs,
         per_joint_ori_errs_pa) = eval_batch(batch)
        all_per_joint_errs.append(1000 * per_joint_errs.numpy())
        all_per_joint_errs_rootrel.append(1000 * per_joint_errs_rootrel.numpy())
        all_per_joint_errs_pa.append(1000 * per_joint_errs_pa.numpy())
        all_vertex_errs.append(1000 * vertex_errs.numpy())
        all_vertex_errs_rootrel.append(1000 * vertex_errs_rootrel.numpy())
        all_vertex_errs_pa.append(1000 * vertex_errs_pa.numpy())
        all_per_joint_ori_errs.append(per_joint_ori_errs.numpy())
        all_per_joint_ori_errs_pa.append(per_joint_ori_errs_pa.numpy())

    all_per_joint_errs = np.concatenate(all_per_joint_errs, axis=0)
    all_per_joint_errs_rootrel = np.concatenate(all_per_joint_errs_rootrel, axis=0)
    all_per_joint_errs_pa = np.concatenate(all_per_joint_errs_pa, axis=0)
    all_vertex_errs = np.concatenate(all_vertex_errs, axis=0)
    all_vertex_errs_rootrel = np.concatenate(all_vertex_errs_rootrel, axis=0)
    all_vertex_errs_pa = np.concatenate(all_vertex_errs_pa, axis=0)
    all_per_joint_ori_errs = np.concatenate(all_per_joint_ori_errs, axis=0)
    all_per_joint_ori_errs_pa = np.concatenate(all_per_joint_ori_errs_pa, axis=0)

    mpjpe = np.mean(all_per_joint_errs_rootrel)
    ampjpe = np.mean(all_per_joint_errs)
    mpjpe_pa = np.mean(all_per_joint_errs_pa)
    mve = np.mean(all_vertex_errs_rootrel)
    amve = np.mean(all_vertex_errs)
    mve_pa = np.mean(all_vertex_errs_pa)

    if FLAGS.joint14:
        major_joint_ids = list(range(12))
    else:
        major_joint_ids = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    minor_joint_ids = [x for x in range(all_per_joint_errs.shape[1]) if x not in major_joint_ids]
    major_dist = all_per_joint_errs_rootrel[:, major_joint_ids]
    minor_dist = all_per_joint_errs_rootrel[:, minor_joint_ids]
    major_dist_abs = all_per_joint_errs[:, major_joint_ids]
    major_dist_pa = all_per_joint_errs_pa[:, major_joint_ids]

    ori_joint_ids = [0, 1, 2, 4, 5, 16, 17, 18, 19]
    mpjae = np.rad2deg(np.mean(all_per_joint_ori_errs[:, ori_joint_ids]))
    mpjae_pa = np.rad2deg(np.mean(all_per_joint_ori_errs_pa[:, ori_joint_ids]))

    max_dist_pa = np.max(major_dist_pa, axis=1)
    ncps_auc = np.mean(np.maximum(0, 1 - max_dist_pa / 300)) * 100
    ncps = [np.mean(max_dist_pa / t <= 1) * 100 for t in [50, 75, 100, 125, 150]]

    minor_pck = np.mean(minor_dist / 50 <= 1) * 100
    pck = np.mean(major_dist / 50 <= 1) * 100
    apck = np.mean(major_dist_abs / 150 <= 1) * 100
    auc = np.mean(np.maximum(0, 1 - (np.floor(major_dist / 199 * 50) + 0.5) / 50)) * 100
    result = ('MPJPE & MPJPE_PA & PCK & AUC & NCPS & NCPS-AUC & APCK & AMPJPE & MinorPCK & MVE & '
              'MVE_PA & MPJAE & MPJAE_PA \n')
    result += to_latex(
        [mpjpe, mpjpe_pa, pck, auc, ncps[3], ncps_auc, apck, ampjpe, minor_pck, mve, mve_pa, mpjae,
         mpjae_pa]) + '\n'
    result += to_latex(ncps) + '\n'
    result += str(np.mean(major_dist / 50 <= 1, axis=0) * 100) + '\n'
    result += str(np.mean(major_dist / 100 <= 1, axis=0) * 100) + '\n'
    result += str(np.mean(major_dist / 150 <= 1, axis=0) * 100) + '\n'
    print(result)
    spu.write_file(result, f'{FLAGS.pred_path}/metrics')
    # np.savez(f'{FLAGS.pred_path}/arrays.npz', true=all_true3d, pred=all_pred3d)

    for thresh in [50, 51, 52, 53, 54, 55, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
        print(thresh, str(np.mean(major_dist / thresh <= 1) * 100))

    # metrics_wacv = eval_wacv23(dist, dist_aligned, dist_abs)
    # print(to_latex(metrics_wacv))
    metrics_eccv24 = [mpjpe, mpjpe_pa, mve, mve_pa, mpjae, mpjae_pa]
    print(to_latex(metrics_eccv24))


@tf.function
def eval_batch(batch):
    is_male = batch['true_gender'] == 'male'
    # print(is_male)
    i_male = tf.where(is_male)[:, 0]
    i_female = tf.where(tf.logical_not(is_male))[:, 0]

    bm_male = smpl.tensorflow.SMPL(model_name='smpl', gender='male')
    bm_female = smpl.tensorflow.SMPL(model_name='smpl', gender='female')

    true_result_male = bm_male(
        pose_rotvecs=tf.gather(batch['true_pose'], i_male, axis=0),
        shape_betas=tf.gather(batch['true_betas'], i_male, axis=0),
        trans=tf.gather(batch['true_trans'], i_male, axis=0))

    true_result_female = bm_female(
        pose_rotvecs=tf.gather(batch['true_pose'], i_female, axis=0),
        shape_betas=tf.gather(batch['true_betas'], i_female, axis=0),
        trans=tf.gather(batch['true_trans'], i_female, axis=0))

    inv_perm = tf.math.invert_permutation(tf.concat([i_male, i_female], axis=0))
    true_result = tf.nest.map_structure(
        lambda x, y: tf.gather(tf.concat([x, y], axis=0), inv_perm),
        true_result_male, true_result_female)

    bm_neutral = smpl.tensorflow.SMPL(model_name='smpl', gender='neutral')
    pred_result = bm_neutral(
        pose_rotvecs=batch['pred_pose'],
        shape_betas=batch['pred_betas'],
        trans=batch['pred_trans'])

    true_joints, true_pelvis = get_joints_and_pelvis(true_result['joints'], true_result['vertices'])
    pred_joints, pred_pelvis = get_joints_and_pelvis(pred_result['joints'], pred_result['vertices'])

    meanY, T, output_scale, meanX = procrustes_tf.procrustes_tf_transf(
        true_joints, pred_joints,
        tf.ones_like(true_joints[..., 0], dtype=tf.bool),
        allow_scaling=True, allow_reflection=False)
    pred_result['joints_aligned'] = ((pred_joints - meanY) @ T) * output_scale + meanX
    pred_result['vertices_aligned'] = ((pred_result['vertices'] - meanY) @ T) * output_scale + meanX
    pred_result['orientations_aligned'] = tf.linalg.matrix_transpose(T)[:, np.newaxis] @ \
                                          pred_result['orientations']

    per_joint_errs = tf.linalg.norm(true_joints - pred_joints, axis=-1)

    per_joint_errs_rootrel = tf.linalg.norm(
        (true_joints - true_pelvis) -
        (pred_joints - pred_pelvis), axis=-1)
    per_joint_errs_pa = tf.linalg.norm(
        true_joints - pred_result['joints_aligned'], axis=-1)
    vertex_errs = tf.linalg.norm(true_result['vertices'] - pred_result['vertices'], axis=-1)
    vertex_errs_rootrel = tf.linalg.norm(
        (true_result['vertices'] - true_pelvis) -
        (pred_result['vertices'] - pred_pelvis), axis=-1)
    vertex_errs_pa = tf.linalg.norm(
        true_result['vertices'] - pred_result['vertices_aligned'], axis=-1)

    per_joint_ori_errs = geodesic_distance_so3(
        true_result['orientations'], pred_result['orientations'])
    per_joint_ori_errs_pa = geodesic_distance_so3(
        true_result['orientations'], pred_result['orientations_aligned'])

    return (
        per_joint_errs, per_joint_errs_rootrel, per_joint_errs_pa, vertex_errs, vertex_errs_rootrel,
        vertex_errs_pa, per_joint_ori_errs, per_joint_ori_errs_pa)


def get_joints_and_pelvis(joints, vertices):
    if FLAGS.joint14:
        if FLAGS.use_h36m_regressor:
            regressor17 = tf.constant(
                np.load(f'{DATA_ROOT}/body_models/J_regressor_h36m.npy'), tf.float32)  # [17, 6890]
            regressor14 = tf.gather(
                regressor17, [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10], axis=0)
            true_joints = regressor14 @ vertices
        else:
            true_joints = tf.gather(
                joints, [1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21], axis=1)

        if FLAGS.align_at_hips:
            true_pelvis = tf.reduce_mean(true_joints[:, 2:4], axis=1, keepdims=True)
        elif FLAGS.use_h36m_regressor:
            true_pelvis = regressor17[:1] @ vertices
        else:
            true_pelvis = true_joints[:, :1]
    else:
        true_joints = joints
        if FLAGS.align_at_hips:
            true_pelvis = tf.reduce_mean(true_joints[:, 1:3], axis=1, keepdims=True)
        else:
            true_pelvis = true_joints[:, :1]

    return true_joints, true_pelvis


def to_latex(numbers):
    return ' & '.join([f'{x:.1f}' if x else '' for x in numbers])


def haversin_rot(rot1, rot2):
    # the haversine is equal to (1-cos(x))/2 = sin(x/2)**2
    # It turns out that the haversine of the angle between two rotations
    # (ie the angle of the axis-angle representation) is just sum of squared differences
    # in 3x3 rotation matrix representation divided by 8.
    return tf.reduce_sum(tf.math.squared_difference(rot1, rot2), axis=(-2, -1)) * 0.125


def arc_haversin(h):
    return 2 * tf.math.asin(tf.minimum(tf.sqrt(h), 1))


def geodesic_distance_so3(rot1, rot2):
    return arc_haversin(haversin_rot(rot1, rot2))


def get_all_gt_poses(testset_only):
    all_valid_pose = {}
    all_valid_betas = {}
    all_valid_trans = {}
    all_valid_gender = {}

    if testset_only:
        seq_filepaths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dpw/sequenceFiles/test/*.pkl')
    else:
        seq_filepaths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')

    for filepath in seq_filepaths:
        seq = spu.load_pickle(filepath)
        seq_name = seq['sequence']
        extrinsics_per_frame = seq['cam_poses']

        for i_person, (
                gender, poses_seq, betas, trans_seq, coords2d_seq, camvalid_seq) in enumerate(
            zip(
                seq['genders'], seq['poses'], seq['betas'], seq['trans'], seq['poses2d'],
                seq['campose_valid'])):
            betas = betas[:10]
            gender = 'male' if gender.startswith('m') else 'female'
            body_model = smpl.numpy.get_cached_body_model('smpl', gender)

            for i_frame, (pose, trans, coords2d, extrinsics, campose_valid) in enumerate(
                    zip(poses_seq, trans_seq, coords2d_seq, extrinsics_per_frame,
                        camvalid_seq)):
                if not campose_valid or np.all(coords2d == 0):
                    continue

                image_relpath = f'imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                pose, trans = body_model.rototranslate(
                    extrinsics[:3, :3], extrinsics[:3, 3], pose, betas, trans)
                all_valid_pose[image_relpath, i_person] = pose
                all_valid_betas[image_relpath, i_person] = betas
                all_valid_trans[image_relpath, i_person] = trans
                all_valid_gender[image_relpath, i_person] = gender

    return all_valid_pose, all_valid_betas, all_valid_trans, all_valid_gender


def get_all_pred_poses():
    if FLAGS.testset_only:
        pred_filepaths = spu.sorted_recursive_glob(f'{FLAGS.pred_path}/test/*.pkl')
    else:
        pred_filepaths = spu.sorted_recursive_glob(f'{FLAGS.pred_path}/**/*.pkl')

    all_pred_pose = {}
    all_pred_betas = {}
    all_pred_trans = {}

    for filepath in pred_filepaths:
        seq_name = osp.splitext(osp.basename(filepath))[0]
        preds = spu.load_pickle(filepath)

        for i_person, (person_pose, person_betas, person_trans) in enumerate(
                zip(preds['pose'], preds['betas'], preds['trans'])):
            for i_frame, (pose, betas, trans) in enumerate(
                    zip(person_pose, person_betas, person_trans)):
                image_relpath = f'imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                all_pred_pose[image_relpath, i_person] = pose
                all_pred_betas[image_relpath, i_person] = betas
                all_pred_trans[image_relpath, i_person] = trans

    return all_pred_pose, all_pred_betas, all_pred_trans


def causal_smooth(tracks):
    kernel = np.array([6, 2, 1, 1, 0.5], np.float32)
    kernel /= np.sum(kernel)
    return scipy.ndimage.convolve1d(tracks, kernel, axis=1, origin=-1, mode='reflect')


def acausal_smooth(tracks):
    # Exponentially decaying kernel (like exponential moving average, but looks at the future, too)
    kernel = np.array([1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1], np.float32)
    kernel /= np.sum(kernel)
    return scipy.ndimage.convolve1d(tracks, kernel, axis=1, mode='reflect')


def eval_wacv23(dist, dist_pa, dist_abs):
    worst_dist = np.max(dist, axis=-1)

    return np.array(
        [np.mean(dist),
         np.mean(dist_pa),
         np.mean(dist_abs),
         # PCK all
         get_pck(dist / 50),
         get_pck(dist / 100),
         get_pck(dist / 150),
         get_auc_real(dist / 50),
         get_auc_real(dist / 100),
         get_auc_real(dist / 150),
         # CPS all
         get_pck(worst_dist / 100),
         get_pck(worst_dist / 150),
         get_pck(worst_dist / 200),
         get_auc_real(worst_dist / 100),
         get_auc_real(worst_dist / 150),
         get_auc_real(worst_dist / 200),
         # APCK all
         get_pck(dist_abs / 150),
         get_pck(dist_abs / 300),
         get_auc_real(dist_abs / 150),
         get_auc_real(dist_abs / 300),
         ])


def get_pck(rel_dists):
    return np.mean(rel_dists < 1) * 100


def get_auc_real(rel_dists):
    return np.mean(np.maximum(0, 1 - rel_dists)) * 100


if __name__ == "__main__":
    main()
