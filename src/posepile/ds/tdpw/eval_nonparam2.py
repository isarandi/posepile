import argparse
import functools
import os.path as osp

import numpy as np
import simplepyutils as spu
import tensorflow as tf
from simplepyutils import FLAGS

import posepile.util.procrustes_tf as procrustes_tf
import smpl.numpy
import smpl.tensorflow
from posepile.ds.tdpw.eval import get_all_gt_poses, get_joints_and_pelvis, to_latex
from posepile.ds.tdpw.stats_collector import StatCollector
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--batch-size', type=str, default=64)
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
             i_persons=np.array(i_persons))).batch(FLAGS.batch_size).prefetch(1)

    stats = StatCollector(
        prod_key_pairs=[
            ('per_joint_errs_rootrel', 'joint_uncert'),
            ('vertex_errs_rootrel', 'vertex_uncert')],
        ssq_keys=[
            'per_joint_errs_rootrel', 'vertex_errs_rootrel',
            'joint_uncert', 'vertex_uncert'])

    for batch in spu.progressbar(ds, total=len(relpaths), step=FLAGS.batch_size):
        r = eval_batch(batch)
        r = tf.nest.map_structure(lambda x: 1000 * x.numpy(), r)
        r['per_joint_pcks_rootrel'] = np.float32(r['per_joint_errs_rootrel'] / 50 <= 1) * 100
        r['per_joint_aucs_rootrel'] = np.maximum(
            0, 1 - (np.floor(r['per_joint_errs_rootrel'] / 199 * 50) + 0.5) / 50) * 100
        r['per_joint_errs_rootrel__joint_uncert__corr'] = corrcoef_pearson(
            r['per_joint_errs_rootrel'], r['joint_uncert'], axis=-1)
        r['vertex_errs_rootrel__vertex_uncert__corr'] = corrcoef_pearson(
            r['vertex_errs_rootrel'], r['vertex_uncert'], axis=-1)
        stats.update(r)

    mpjpe = stats.get_mean('per_joint_errs_rootrel', -1)
    ampjpe = stats.get_mean('per_joint_errs', -1)
    mpjpe_pa = stats.get_mean('per_joint_errs_pa', -1)
    mve = stats.get_mean('vertex_errs_rootrel', -1)
    amve = stats.get_mean('vertex_errs', -1)
    mve_pa = stats.get_mean('vertex_errs_pa', -1)
    print(mpjpe, ampjpe, mpjpe_pa, mve, amve, mve_pa)

    if not FLAGS.joint14:
        pcc_joints = stats.get_corr('per_joint_errs_rootrel', 'joint_uncert', -1) * 100
        pcc_vertices = stats.get_corr('vertex_errs_rootrel', 'vertex_uncert', -1) * 100
        # pcc_joints = stats.get_mean('per_joint_errs_rootrel__joint_uncert__corr') * 100
        # pcc_vertices = stats.get_mean('vertex_errs_rootrel__vertex_uncert__corr') * 100
        average_uncert_joint = stats.get_mean('joint_uncert', -1)
        average_uncert_vertex = stats.get_mean('vertex_uncert', -1)
        sem_uncert_joint = stats.get_sem('joint_uncert', -1)
        sem_uncert_vertex = stats.get_sem('vertex_uncert', -1)
        sem_joint_errs_rootrel = stats.get_sem('per_joint_errs_rootrel', -1)
        sem_vertex_errs_rootrel = stats.get_sem('vertex_errs_rootrel', -1)
    else:
        pcc_joints = None
        pcc_vertices = None
        average_uncert_joint = None
        average_uncert_vertex = None
        sem_uncert_joint = None
        sem_uncert_vertex = None
        sem_joint_errs_rootrel = None
        sem_vertex_errs_rootrel = None

    if FLAGS.joint14:
        major_joint_ids = list(range(12))
    else:
        major_joint_ids = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]

    n_joints = stats.get_sum('per_joint_errs_rootrel').shape[0]

    minor_joint_ids = [x for x in range(n_joints) if x not in major_joint_ids]

    pck = np.mean(stats.get_mean('per_joint_pcks_rootrel')[major_joint_ids])
    # minor_dist = all_eval_results['per_joint_errs_rootrel'][:, minor_joint_ids]
    # major_dist_abs = all_eval_results['per_joint_errs'][:, major_joint_ids]
    # major_dist_pa = all_eval_results['per_joint_errs_pa'][:, major_joint_ids]
    # max_dist_pa = np.max(major_dist_pa, axis=1)
    # ncps_auc = np.mean(np.maximum(0, 1 - max_dist_pa / 300)) * 100
    # ncps = [np.mean(max_dist_pa / t <= 1) * 100 for t in [50, 75, 100, 125, 150]]
    #
    # minor_pck = np.mean(minor_dist / 50 <= 1) * 100
    # apck = np.mean(major_dist_abs / 150 <= 1) * 100
    auc = np.mean(stats.get_mean('per_joint_aucs_rootrel')[major_joint_ids])
    result = ('MPJPE & MPJPE_PA & '
              'PCK & AUC & '
              'AMPJPE & '
              'MVE & MVE_PA & '
              'SDJE & SDVE & '
              'MJU & MVU & '
              'SDJU & SDVU & '
              'PCCJ & PCCV \n')
    result += to_latex(
        [mpjpe, mpjpe_pa,
         pck, auc,
         ampjpe,
         mve, mve_pa,
         sem_joint_errs_rootrel, sem_vertex_errs_rootrel,
         average_uncert_joint, average_uncert_vertex,
         sem_uncert_joint, sem_uncert_vertex,
         pcc_joints, pcc_vertices
         ]) + '\n'
    # result += to_latex(ncps) + '\n'
    # result += str(np.mean(major_dist / 50 <= 1, axis=0) * 100) + '\n'
    # result += str(np.mean(major_dist / 100 <= 1, axis=0) * 100) + '\n'
    # result += str(np.mean(major_dist / 150 <= 1, axis=0) * 100) + '\n'
    print(result)
    # spu.write_file(result, f'{FLAGS.pred_path}/metrics')
    #
    # for thresh in [50, 51, 52, 53, 54, 55, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
    #     print(thresh, str(np.mean(major_dist / thresh <= 1) * 100))

    # metrics_eccv24 = [mpjpe, mpjpe_pa, mve, mve_pa, '', '']
    # print(to_latex(metrics_eccv24))


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
    # batch should have keys: true_gender, true_pose, true_betas, true_trans
    # pred_result should have keys: joints, vertices (with fourth channel being the uncert)
    is_male = batch['true_gender'] == 'male'
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
    pred_joints, pred_pelvis = get_joints_and_pelvis(
        pred_result['joints'][..., :3], pred_result['vertices'][..., :3])

    if pred_result['joints'].shape[-1] == 4:
        joint_uncert = pred_result['joints'][..., 3]
        vertex_uncert = pred_result['vertices'][..., 3]
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
    pred_result['vertices_aligned'] = (
            ((pred_result['vertices'][..., :3] - meanY) @ T) * output_scale + meanX)

    per_joint_errs = tf.linalg.norm(true_joints - pred_joints, axis=-1)
    per_joint_errs_rootrel = tf.linalg.norm(
        (true_joints - true_pelvis) -
        (pred_joints - pred_pelvis), axis=-1)
    per_joint_errs_pa = tf.linalg.norm(true_joints - pred_result['joints_aligned'], axis=-1)
    vertex_errs = tf.linalg.norm(
        true_result['vertices'] - pred_result['vertices'][..., :3], axis=-1)
    vertex_errs_rootrel = tf.linalg.norm(
        (true_result['vertices'] - true_pelvis) -
        (pred_result['vertices'][..., :3] - pred_pelvis), axis=-1)
    vertex_errs_pa = tf.linalg.norm(
        true_result['vertices'] - pred_result['vertices_aligned'], axis=-1)

    return dict(
        per_joint_errs=per_joint_errs, per_joint_errs_rootrel=per_joint_errs_rootrel,
        per_joint_errs_pa=per_joint_errs_pa, vertex_errs=vertex_errs,
        vertex_errs_rootrel=vertex_errs_rootrel, vertex_errs_pa=vertex_errs_pa,
        joint_uncert=joint_uncert, vertex_uncert=vertex_uncert)


@functools.lru_cache()
def get_subdir_mapping():
    seq_filepaths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')
    result = {}
    for filepath in seq_filepaths:
        seq_name = osp.splitext(osp.basename(filepath))[0]
        phase = spu.split_path(filepath)[-2]
        result[seq_name] = phase
    return result


from humcentr_cli.util.serialization import Reader


def corrcoef_pearson(x, y, axis=None):
    x = x - np.mean(x, axis=axis, keepdims=True)
    y = y - np.mean(y, axis=axis, keepdims=True)
    return np.sum(x * y, axis=axis) / np.sqrt(np.sum(x ** 2, axis=axis) * np.sum(y ** 2, axis=axis))


def corrcoef_spearman(x, y, axis=None):
    x_rank = np.argsort(np.argsort(x, axis=axis), axis=axis)
    y_rank = np.argsort(np.argsort(y, axis=axis), axis=axis)
    return corrcoef_pearson(x_rank, y_rank, axis=axis)


@functools.lru_cache(maxsize=3)
def load_seq_predfile(path):
    with Reader(path) as reader:
        dict_per_frame = list(reader)
    return dict_per_frame


def load_pred(image_path, i_person):
    seq_name = spu.split_path(image_path)[-2]
    i_frame = int(osp.splitext(osp.basename(image_path))[0].split('_')[-1])
    phase = get_subdir_mapping()[seq_name]
    preds = load_seq_predfile(f'{FLAGS.pred_path}/{phase}/{seq_name}.xz')
    jp = preds[i_frame]['joints'][i_person] / 1000
    vp = preds[i_frame]['vertices'][i_person] / 1000
    ju = preds[i_frame]['joint_uncertainties'][i_person]
    vu = preds[i_frame]['vertex_uncertainties'][i_person]
    j = np.concatenate([jp, ju[..., np.newaxis]], axis=-1)
    v = np.concatenate([vp, vu[..., np.newaxis]], axis=-1)
    return dict(joints=j, vertices=v)


if __name__ == "__main__":
    main()
