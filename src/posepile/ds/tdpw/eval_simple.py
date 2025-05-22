import argparse
import glob
import os.path as osp

import cameravision
import numpy as np
import scipy.ndimage
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.util.rigid_alignment as rigid_alignment
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--procrustes', action=spu.argparse.BoolAction)
    parser.add_argument('--testset-only', action=spu.argparse.BoolAction)
    parser.add_argument('--acausal-smoothing', action=spu.argparse.BoolAction)
    parser.add_argument('--causal-smoothing', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)

    poses3d_true_dict = get_all_gt_poses()
    poses3d_pred_dict = get_all_pred_poses()

    all_true3d_abs = np.array(list(poses3d_true_dict.values()))
    all_pred3d_abs = np.array([poses3d_pred_dict[relpath] for relpath in poses3d_true_dict])
    all_true3d = all_true3d_abs - all_true3d_abs[:, :1]
    all_pred3d = all_pred3d_abs - all_pred3d_abs[:, :1]
    all_pred3d_aligned = rigid_alignment.rigid_align_many(
        all_pred3d, all_true3d, scale_align=True)

    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    dist_abs = np.linalg.norm(all_true3d_abs - all_pred3d_abs, axis=-1)
    dist_aligned = np.linalg.norm(all_true3d - all_pred3d_aligned, axis=-1)

    mpjpe = np.mean(dist)
    ampjpe = np.mean(dist_abs)
    mpjpe_pa = np.mean(dist_aligned)
    major_joint_ids = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    minor_joint_ids = [x for x in range(dist.shape[1]) if x not in major_joint_ids]
    major_dist = dist[:, major_joint_ids]
    minor_dist = dist[:, minor_joint_ids]
    major_dist_abs = dist_abs[:, major_joint_ids]
    major_dist_pa = dist_aligned[:, major_joint_ids]

    max_dist_pa = np.max(major_dist_pa, axis=1)
    ncps_auc = np.mean(np.maximum(0, 1 - max_dist_pa / 300)) * 100
    ncps = [np.mean(max_dist_pa / t <= 1) * 100 for t in [50, 75, 100, 125, 150]]

    minor_pck = np.mean(minor_dist / 50 <= 1) * 100
    pck = np.mean(major_dist / 50 <= 1) * 100
    apck = np.mean(major_dist_abs / 150 <= 1) * 100
    auc = np.mean(np.maximum(0, 1 - (np.floor(major_dist / 199 * 50) + 0.5) / 50)) * 100
    result = 'MPJPE & MPJPE_PA & PCK & AUC & NCPS & NCPS-AUC & APCK & AMPJPE & MinorPCK \n'
    result += to_latex(
        [mpjpe, mpjpe_pa, pck, auc, ncps[3], ncps_auc, apck, ampjpe, minor_pck]) + '\n'
    result += to_latex(ncps) + '\n'
    result += str(np.mean(major_dist / 50 <= 1, axis=0) * 100) + '\n'
    result += str(np.mean(major_dist / 100 <= 1, axis=0) * 100) + '\n'
    result += str(np.mean(major_dist / 150 <= 1, axis=0) * 100) + '\n'
    print(result)
    spu.write_file(result, f'{FLAGS.pred_path}/metrics')
    np.savez(f'{FLAGS.pred_path}/arrays.npz', true=all_true3d, pred=all_pred3d)

    for thresh in [50, 51, 52, 53, 54, 55, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
        print(thresh, str(np.mean(major_dist / thresh <= 1) * 100))

    metrics_wacv = eval_wacv23(dist, dist_aligned, dist_abs)
    print(to_latex(metrics_wacv))


def to_latex(numbers):
    return ' & '.join([f'{x:.5f}' for x in numbers])


def get_all_gt_poses():
    all_valid_poses = {}
    if FLAGS.testset_only:
        seq_filepaths = glob.glob(f'{DATA_ROOT}/3dpw/sequenceFiles/test/*.pkl')
    else:
        seq_filepaths = glob.glob(f'{DATA_ROOT}/3dpw/sequenceFiles/*/*.pkl')

    for filepath in seq_filepaths:
        seq = spu.load_pickle(filepath)
        seq_name = seq['sequence']
        intrinsics = seq['cam_intrinsics']
        extrinsics_per_frame = seq['cam_poses']

        for i_person, (coord3d_seq, coords2d_seq, camvalid_seq) in enumerate(zip(
                seq['jointPositions'], seq['poses2d'], seq['campose_valid'])):
            for i_frame, (coords3d, coords2d, extrinsics, campose_valid) in enumerate(
                    zip(coord3d_seq, coords2d_seq, extrinsics_per_frame, camvalid_seq)):
                if not campose_valid or np.all(coords2d == 0):
                    continue
                image_relpath = f'imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                camera = cameravision.Camera(
                    extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
                    world_up=(0, 1, 0))
                camera.t *= 1000
                world_coords = coords3d.reshape(-1, 3) * 1000
                camcoords = camera.world_to_camera(world_coords)
                all_valid_poses[image_relpath, i_person] = camcoords

    return all_valid_poses


def get_all_pred_poses():
    if FLAGS.testset_only:
        pred_filepaths = glob.glob(f'{FLAGS.pred_path}/test/*.pkl', recursive=True)
    else:
        pred_filepaths = glob.glob(f'{FLAGS.pred_path}/**/*.pkl', recursive=True)

    all_pred_poses = {}
    for filepath in pred_filepaths:
        seq_name = osp.splitext(osp.basename(filepath))[0]
        preds = spu.load_pickle(filepath)['jointPositions']
        if FLAGS.causal_smoothing:
            preds = causal_smooth(preds)
        elif FLAGS.acausal_smoothing:
            preds = acausal_smooth(preds)
        for i_person, person_preds in enumerate(preds):
            for i_frame, pred in enumerate(person_preds):
                image_relpath = f'imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                all_pred_poses[image_relpath, i_person] = pred * 1000

    return all_pred_poses


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
