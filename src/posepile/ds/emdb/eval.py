import argparse
import collections
import os.path as osp

import numpy as np
import simplepyutils as spu
import smpl.numpy
import smpl.tensorflow
import tensorflow as tf
from posepile.ds.tdpw.eval import eval_batch, to_latex
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, required=True)
    parser.add_argument('--num-betas', type=int, default=10)
    parser.add_argument('--align-at-hips', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--joint14', action=spu.argparse.BoolAction)
    parser.add_argument('--use-h36m-regressor', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)

    true_pose_dict, true_betas_dict, true_trans_dict, true_gender_dict = get_all_gt_poses()
    pred_pose_dict, pred_betas_dict, pred_trans_dict = get_all_pred_poses()
    all_true_pose = np.array(list(true_pose_dict.values()))
    all_true_betas = np.array(list(true_betas_dict.values()))
    all_true_trans = np.array(list(true_trans_dict.values()))
    all_true_gender = np.array(list(true_gender_dict.values()))

    relpaths = list(true_pose_dict.keys())
    all_pred_pose = np.array([pred_pose_dict[relpath] for relpath in relpaths])
    all_pred_betas = np.array([pred_betas_dict[relpath] for relpath in relpaths])
    all_pred_trans = np.array([pred_trans_dict[relpath] for relpath in relpaths])

    ds = tf.data.Dataset.from_tensor_slices(
        dict(true_pose=all_true_pose, true_betas=all_true_betas, true_trans=all_true_trans,
             true_gender=all_true_gender, pred_pose=all_pred_pose, pred_betas=all_pred_betas,
             pred_trans=all_pred_trans)).batch(64)

    all_per_joint_errs = []
    all_per_joint_errs_rootrel = []
    all_per_joint_errs_pa = []
    all_vertex_errs = []
    all_vertex_errs_rootrel = []
    all_vertex_errs_pa = []
    all_per_joint_ori_errs = []
    all_per_joint_ori_errs_pa = []

    for batch in spu.progressbar(ds, total=len(relpaths) // 64):
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
    # spu.write_file(result, f'{FLAGS.pred_path}/metrics')
    # np.savez(f'{FLAGS.pred_path}/arrays.npz', true=all_true3d, pred=all_pred3d)

    for thresh in [50, 51, 52, 53, 54, 55, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
        print(thresh, str(np.mean(major_dist / thresh <= 1) * 100))

    # metrics_wacv = eval_wacv23(dist, dist_aligned, dist_abs)
    # print(to_latex(metrics_wacv))



def get_all_gt_poses():
    all_valid_pose = {}
    all_valid_betas = {}
    all_valid_trans = {}
    all_valid_gender = {}

    all_emdb_pkl_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/emdb/**/*_data.pkl')
    emdb1_sequence_roots = [
        osp.dirname(p) for p in all_emdb_pkl_paths
        if spu.load_pickle(p)['emdb1']]

    results_all = collections.defaultdict(list)

    for seq_root in emdb1_sequence_roots:
        seq_name = osp.basename(seq_root)
        logger.info(f'Predicting {seq_name}...')
        subj = seq_root.split('/')[-2]
        seq_data = spu.load_pickle(f'{seq_root}/{subj}_{seq_name}_data.pkl')

        gender = seq_data["gender"]
        poses_body = seq_data["smpl"]["poses_body"]
        poses_root = seq_data["smpl"]["poses_root"]

        betas = seq_data["smpl"]["betas"]
        trans_seq = seq_data["smpl"]["trans"]
        poses_seq = np.concatenate([poses_root, poses_body], axis=-1)
        is_valid_seq = seq_data['good_frames_mask']
        extrinsics_seq = seq_data['camera']['extrinsics']

        body_model = smpl.numpy.get_cached_body_model('smpl', gender)

        for i_frame, (pose, trans, extrinsics, is_valid) in enumerate(
                zip(poses_seq, trans_seq, extrinsics_seq, is_valid_seq)):
            if not is_valid:
                continue
            image_relpath = f'{subj}/{seq_name}/images/{i_frame:05d}.jpg'

            pose, trans = body_model.rototranslate(
                extrinsics[:3, :3], extrinsics[:3, 3], pose, betas, trans)

            all_valid_pose[image_relpath] = pose
            all_valid_betas[image_relpath] = betas
            all_valid_trans[image_relpath] = trans
            all_valid_gender[image_relpath] = gender

    return all_valid_pose, all_valid_betas, all_valid_trans, all_valid_gender


def get_all_pred_poses():
    preds = spu.load_pickle(FLAGS.pred_path)
    all_pred_pose = {}
    all_pred_betas = {}
    all_pred_trans = {}

    for seq_id, pred in preds.items():
        subject_id, seq_name = seq_id.split('_', maxsplit=1)
        for i_frame, (pose, betas, trans) in enumerate(
                zip(pred['pose'], pred['betas'], pred['trans'])):
            image_relpath = f'{subject_id}/{seq_name}/images/{i_frame:05d}.jpg'
            all_pred_pose[image_relpath] = pose
            all_pred_betas[image_relpath] = betas[:FLAGS.num_betas]
            all_pred_trans[image_relpath] = trans

    return all_pred_pose, all_pred_betas, all_pred_trans


if __name__ == "__main__":
    main()
