import argparse
import re

import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.ds.h36m.main as h36m
import posepile.util.rigid_alignment as rigid_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--procrustes', action=spu.argparse.BoolAction)
    parser.add_argument('--only-S11', action=spu.argparse.BoolAction)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument(
        '--root-last', action=spu.argparse.BoolAction, help='Use last joint as root')
    spu.argparse.initialize(parser)

    all_image_relpaths, all_true3d = get_all_gt_poses()
    activities = np.array(
        [re.search(f'Images/(.+?)\.', path)[1].split(' ')[0]
         for path in all_image_relpaths])

    if FLAGS.seeds > 1:
        mean_per_seed, std_per_seed = evaluate_multiple_seeds(all_true3d, activities)
        print(to_latex(mean_per_seed))
        print(to_latex(std_per_seed))
    else:
        metrics = evaluate(FLAGS.pred_path, all_true3d, activities)
        print(to_latex(metrics))

        metrics_wacv = eval_wacv23(FLAGS.pred_path)
        print(to_latex(metrics_wacv))


def evaluate_multiple_seeds(all_true3d, activities):
    seed_pred_paths = [FLAGS.pred_path.replace('seed1', f'seed{i + 1}') for i in range(FLAGS.seeds)]
    metrics_per_seed = np.array([evaluate(p, all_true3d, activities) for p in seed_pred_paths])
    mean_per_seed = np.mean(metrics_per_seed, axis=0)
    std_per_seed = np.std(metrics_per_seed, axis=0)
    return mean_per_seed, std_per_seed


def evaluate(pred_path, all_true3d, activities):
    all_pred3d = get_all_pred_poses(pred_path)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    i_root = -1 if FLAGS.root_last else 0
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    all_true3d -= all_true3d[:, i_root, np.newaxis]

    ordered_activities = (
            'Directions Discussion Eating Greeting Phoning Posing Purchases ' +
            'Sitting SittingDown Smoking Photo Waiting Walking WalkDog WalkTogether').split()
    if FLAGS.procrustes:
        all_pred3d = rigid_alignment.rigid_align_many(all_pred3d, all_true3d, scale_align=True)
    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    overall_mean_error = np.mean(dist)
    # The overall mean here is computed across all poses of the dataset
    metrics = [np.mean(dist[activities == activity]) for activity in ordered_activities]
    metrics.append(overall_mean_error)
    return metrics


def to_latex(numbers):
    return ' & '.join([f'{x:.1f}' for x in numbers])


def get_all_gt_poses(i_relevant_joints=None):
    if i_relevant_joints is None:
        if FLAGS.root_last:
            i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
        else:
            i_relevant_joints = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    return h36m.get_all_gt_poses(
        i_subjects=[11] if FLAGS.only_S11 else [9, 11], i_relevant_joints=i_relevant_joints,
        frame_step=64)


def get_all_pred_poses(path, n_joints=17):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_path'])
    image_paths = results['image_path'][order]
    j_select_17_from_25 = [24, 0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21]
    poses_pred = results['coords3d_pred_world']
    if n_joints == 17 and poses_pred.shape[1] == 25:
        poses_pred = poses_pred[:, j_select_17_from_25]

    if FLAGS.only_S11:
        needed = ['S11' in p for p in image_paths]
        return poses_pred[order][needed]

    return poses_pred[order]


def eval_wacv23(pred_path):
    all_image_relpaths, all_true3d = get_all_gt_poses()
    # i_relevant_joints=[
    #    *range(1, 11), 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30, 0])

    all_pred3d = get_all_pred_poses(pred_path, n_joints=17)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    i_root = -1 if FLAGS.root_last else 0
    all_pred3d_abs = all_pred3d.copy()
    all_true3d_abs = all_true3d.copy()
    all_pred3d -= all_pred3d[:, i_root, np.newaxis]
    all_true3d -= all_true3d[:, i_root, np.newaxis]

    all_pred3d_pa = rigid_alignment.rigid_align_many(all_pred3d, all_true3d, scale_align=True)

    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    dist_pa = np.linalg.norm(all_true3d - all_pred3d_pa, axis=-1)
    dist_abs = np.linalg.norm(all_true3d_abs - all_pred3d_abs, axis=-1)
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


if __name__ == '__main__':
    main()
