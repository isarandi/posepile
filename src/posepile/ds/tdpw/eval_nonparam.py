import argparse
import os.path as osp

import numpy as np
import posepile.util.procrustes_tf as procrustes_tf
import scipy.ndimage
import simplepyutils as spu
import smpl.numpy
import smpl.tensorflow
import tensorflow as tf
import scipy.stats
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS
import functools
from posepile.ds.tdpw.eval import get_joints_and_pelvis


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
    all_true_pose = np.array(list(true_pose_dict.values()))
    all_true_betas = np.array(list(true_betas_dict.values()))
    all_true_trans = np.array(list(true_trans_dict.values()))
    all_true_gender = np.array(list(true_gender_dict.values()))

    relpaths = list(true_pose_dict.keys())
    impaths, i_persons = zip(*relpaths)
    ds = tf.data.Dataset.from_tensor_slices(
        dict(true_pose=all_true_pose, true_betas=all_true_betas, true_trans=all_true_trans,
             true_gender=all_true_gender, impaths=np.array(impaths),
             i_persons=np.array(i_persons))).batch(64).prefetch(1)

    all_per_joint_errs = []
    all_per_joint_errs_rootrel = []
    all_per_joint_errs_pa = []
    all_vertex_errs = []
    all_vertex_errs_rootrel = []
    all_vertex_errs_pa = []
    all_joint_uncert = []
    all_vertex_uncert = []

    for batch in spu.progressbar(ds, total=len(relpaths), step=64):
        (per_joint_errs, per_joint_errs_rootrel, per_joint_errs_pa, vertex_errs,
         vertex_errs_rootrel, vertex_errs_pa, joint_uncert, vertex_uncert) = eval_batch(batch)
        all_per_joint_errs.append(1000 * per_joint_errs.numpy())
        all_per_joint_errs_rootrel.append(1000 * per_joint_errs_rootrel.numpy())
        all_per_joint_errs_pa.append(1000 * per_joint_errs_pa.numpy())
        all_vertex_errs.append(1000 * vertex_errs.numpy())
        all_vertex_errs_rootrel.append(1000 * vertex_errs_rootrel.numpy())
        all_vertex_errs_pa.append(1000 * vertex_errs_pa.numpy())
        all_joint_uncert.append(1000 * joint_uncert.numpy())
        all_vertex_uncert.append(1000 * vertex_uncert.numpy())

    all_per_joint_errs = np.concatenate(all_per_joint_errs, axis=0)
    all_per_joint_errs_rootrel = np.concatenate(all_per_joint_errs_rootrel, axis=0)
    all_per_joint_errs_pa = np.concatenate(all_per_joint_errs_pa, axis=0)
    all_vertex_errs = np.concatenate(all_vertex_errs, axis=0)
    all_vertex_errs_rootrel = np.concatenate(all_vertex_errs_rootrel, axis=0)
    all_vertex_errs_pa = np.concatenate(all_vertex_errs_pa, axis=0)
    all_joint_uncert = np.concatenate(all_joint_uncert, axis=0)
    all_vertex_uncert = np.concatenate(all_vertex_uncert, axis=0)

    mpjpe = np.mean(all_per_joint_errs_rootrel)
    ampjpe = np.mean(all_per_joint_errs)
    mpjpe_pa = np.mean(all_per_joint_errs_pa)
    mve = np.mean(all_vertex_errs_rootrel)
    amve = np.mean(all_vertex_errs)
    mve_pa = np.mean(all_vertex_errs_pa)
    if not FLAGS.joint14:
        pcc_joints = np.corrcoef(all_per_joint_errs_rootrel.flatten(), all_joint_uncert.flatten())[0, 1] * 100
        pcc_vertices = np.corrcoef(all_vertex_errs_rootrel.flatten(), all_vertex_uncert.flatten())[0, 1] * 100
        average_uncert_joint = np.mean(all_joint_uncert)
        average_uncert_vertex = np.mean(all_vertex_uncert)
        std_uncert_joint = scipy.stats.sem(all_joint_uncert, axis=None)
        std_uncert_vertex = scipy.stats.sem(all_vertex_uncert, axis=None)
        stdev_joint_errs_rootrel = scipy.stats.sem(all_per_joint_errs_rootrel, axis=None)
        stdev_vertex_errs_rootrel = scipy.stats.sem(all_vertex_errs_rootrel, axis=None)
    else:
        pcc_joints = None
        pcc_vertices = None
        average_uncert_joint = None
        average_uncert_vertex = None
        std_uncert_joint = None
        std_uncert_vertex = None
        stdev_joint_errs_rootrel = None
        stdev_vertex_errs_rootrel = None

    if FLAGS.joint14:
        major_joint_ids = list(range(12))
    else:
        major_joint_ids = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    minor_joint_ids = [x for x in range(all_per_joint_errs.shape[1]) if x not in major_joint_ids]
    major_dist = all_per_joint_errs_rootrel[:, major_joint_ids]
    minor_dist = all_per_joint_errs_rootrel[:, minor_joint_ids]
    major_dist_abs = all_per_joint_errs[:, major_joint_ids]
    major_dist_pa = all_per_joint_errs_pa[:, major_joint_ids]

    max_dist_pa = np.max(major_dist_pa, axis=1)
    ncps_auc = np.mean(np.maximum(0, 1 - max_dist_pa / 300)) * 100
    ncps = [np.mean(max_dist_pa / t <= 1) * 100 for t in [50, 75, 100, 125, 150]]

    minor_pck = np.mean(minor_dist / 50 <= 1) * 100
    pck = np.mean(major_dist / 50 <= 1) * 100
    apck = np.mean(major_dist_abs / 150 <= 1) * 100
    auc = np.mean(np.maximum(0, 1 - (np.floor(major_dist / 199 * 50) + 0.5) / 50)) * 100
    result = ('MPJPE & MPJPE_PA & '
              'PCK & AUC & '
              'NCPS & NCPS-AUC & '
              'APCK & AMPJPE & '
              'MinorPCK & '
              'MVE & MVE_PA & '
              'SDJE & SDVE & '
              'MJU & MVU & '
              'SDJU & SDVU & '
              'PCCJ & PCCV \n')
    result += to_latex(
        [mpjpe, mpjpe_pa,
         pck, auc,
         ncps[3], ncps_auc,
         apck, ampjpe,
         minor_pck,
         mve, mve_pa,
         stdev_joint_errs_rootrel, stdev_vertex_errs_rootrel,
         average_uncert_joint, average_uncert_vertex,
         std_uncert_joint, std_uncert_vertex,
         pcc_joints, pcc_vertices]) + '\n'
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
    metrics_eccv24 = [mpjpe, mpjpe_pa, mve, mve_pa, '', '']
    print(to_latex(metrics_eccv24))
    # print(to_latex(metrics_wacv))


def eval_batch(batch):
    pred_result = {'joints': [], 'vertices': []}
    for impath, i_person in zip(batch['impaths'], batch['i_persons']):
        impath = impath.numpy().decode('utf-8')
        i_person = int(i_person.numpy())
        pred_result_now = load_pred(impath, i_person)
        pred_result['joints'].append(pred_result_now['joints'])
        pred_result['vertices'].append(pred_result_now['vertices'])

    pred_result['joints'] = tf.stack(pred_result['joints'], axis=0)
    pred_result['vertices'] = tf.stack(pred_result['vertices'], axis=0)
    return eval_batch_tfunc(batch, pred_result)


@tf.function
def eval_batch_tfunc(batch, pred_result):
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
    true_joints, true_pelvis = get_joints_and_pelvis(true_result['joints'], true_result['vertices'])
    pred_joints, pred_pelvis = get_joints_and_pelvis(pred_result['joints'][..., :3], pred_result['vertices'][..., :3])

    if pred_result['joints'].shape[-1] == 4:
        joint_uncert = 1000*pred_result['joints'][..., 3]
        vertex_uncert = 1000*pred_result['vertices'][..., 3]
    else:
        if not FLAGS.joint14:
            tf.print('no uncertainty found')
        joint_uncert = tf.zeros_like(pred_result['joints'][..., 0])
        vertex_uncert = tf.zeros_like(pred_result['vertices'][..., 0])

    meanY, T, output_scale, meanX = procrustes_tf.procrustes_tf_transf(
        true_joints, pred_joints,
        tf.ones_like(true_joints[..., 0], dtype=tf.bool),
        allow_scaling=True, allow_reflection=False)

    pred_result = dict(pred_result)
    pred_result['joints_aligned'] = ((pred_joints - meanY) @ T) * output_scale + meanX
    pred_result['vertices_aligned'] = ((pred_result['vertices'][..., :3] - meanY) @ T) * output_scale + meanX

    per_joint_errs = tf.linalg.norm(
        true_joints - pred_joints, axis=-1)

    per_joint_errs_rootrel = tf.linalg.norm(
        (true_joints - true_pelvis) -
        (pred_joints - pred_pelvis), axis=-1)
    per_joint_errs_pa = tf.linalg.norm(
        true_joints - pred_result['joints_aligned'], axis=-1)
    vertex_errs = tf.linalg.norm(true_result['vertices'] - pred_result['vertices'][..., :3], axis=-1)
    vertex_errs_rootrel = tf.linalg.norm(
        (true_result['vertices'] - true_pelvis) -
        (pred_result['vertices'][..., :3] - pred_pelvis), axis=-1)
    vertex_errs_pa = tf.linalg.norm(
        true_result['vertices'] - pred_result['vertices_aligned'], axis=-1)

    return (
        per_joint_errs, per_joint_errs_rootrel, per_joint_errs_pa, vertex_errs, vertex_errs_rootrel,
        vertex_errs_pa, joint_uncert, vertex_uncert)


@functools.lru_cache()
def get_subdir_mapping():
    seq_filepaths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    result = {}
    for filepath in seq_filepaths:
        seq_name = osp.splitext(osp.basename(filepath))[0]
        phase = spu.split_path(filepath)[-2]
        result[seq_name] = phase
    return result


@functools.lru_cache(maxsize=3)
def load_seq_predfile(path):
    return spu.load_pickle(path)


def load_pred(image_path, i_person):
    seq_name = spu.split_path(image_path)[-2]
    i_frame = int(osp.splitext(osp.basename(image_path))[0].split('_')[-1])
    phase = get_subdir_mapping()[seq_name]
    preds = load_seq_predfile(f'{FLAGS.pred_path}/{phase}/{seq_name}.pkl')
    return dict(
        joints=preds['jointPositions'][i_person][i_frame],
        vertices=preds['vertices'][i_person][i_frame])


def to_latex(numbers):
    print(numbers)
    return ' & '.join([f'{x:.1f}' if x else '' for x in numbers])


@spu.picklecache(f'all_tdpw_gt_poses_.pkl')
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
        intrinsics = seq['cam_intrinsics']
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
