import argparse
import glob
import itertools
import re
import more_itertools
import cameralib
import numpy as np
import simplepyutils as spu
import smpl.numpy
from simplepyutils.argparse import FLAGS

import posepile.util as util
import posepile.util.rigid_alignment as rigid_alignment
from posepile.paths import DATA_ROOT
import smpl.render
import rlemasklib
import smpl.tensorflow.fitting
import smpl.tensorflow.full_fitting
import tensorflow as tf
import smpl.tensorflow


def get_video_person(t):
    pattern = r'(?P<video_name>.+)_clip_(?P<clip_num>\d+)_person_(?P<person_num>\d+)_frame_(?P<frame_num>\d+).png'
    m = re.match(pattern, t)
    return m['video_name'], m['person_num']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, required=True)
    parser.add_argument('--num-betas', type=int, default=10)
    parser.add_argument('--l2-regul', type=float, default=1)
    parser.add_argument('--num-iter', type=int, default=3)
    parser.add_argument('--group-sizes', type=int, default=None)
    spu.argparse.initialize(parser)

    preds = np.load(FLAGS.pred_path)
    labels = np.load(f'{DATA_ROOT}/ssp_3d/labels.npz')
    body_model_name = 'smpl'
    bm = smpl.tensorflow.get_cached_body_model(body_model_name, 'neutral')

    K = np.array([[5000, 0., 512 / 2.0],
                  [0., 5000, 512 / 2.0],
                  [0., 0., 1.]])
    camera = cameralib.Camera(intrinsic_matrix=K, world_up=(0, -1, 0))
    # vertex_subset = np.load(f'{DATA_ROOT}/body_models/smpl/vertex_subset.npy')
    # n_subset = 2048
    # vertex_subset = np.load(f'{DATA_ROOT}/body_models/smpl/vertex_subset_{n_subset}.npz')[
    # 'i_verts']
    vertex_subset = np.arange(bm.num_vertices)
    FLAGS.l2_regul *= (len(vertex_subset) + bm.num_joints) / (6890 + 24)
    fit_fn_scale = smpl.tensorflow.get_fit_fn(
        body_model_name, 'neutral',
        requested_keys=('vertices', 'shape_betas', 'pose_rotvecs'), l2_regularizer2=0,
        num_betas=FLAGS.num_betas, l2_regularizer=FLAGS.l2_regul, num_iter=FLAGS.num_iter,
        vertex_subset=vertex_subset, weighted=True, share_beta=FLAGS.group_sizes is not None,
        scale_fit=True, scale_regularizer=0, final_adjust_rots=False)

    fit_fn = smpl.tensorflow.get_fit_fn(
        body_model_name, 'neutral',
        requested_keys=('vertices', 'shape_betas', 'pose_rotvecs'), l2_regularizer2=0,
        num_betas=FLAGS.num_betas, l2_regularizer=FLAGS.l2_regul, num_iter=FLAGS.num_iter,
        vertex_subset=vertex_subset, weighted=True, share_beta=FLAGS.group_sizes is not None,
        scale_fit=False, final_adjust_rots=True)

    def get_weights(uncerts, axis=-1):
        w = uncerts ** -1.5
        return w / np.mean(w, axis=axis, keepdims=True)

    vertex_weights = get_weights(preds['vertex_uncertainties'])
    joint_weights = get_weights(preds['joint_uncertainties'])
    rng = np.random.Generator(np.random.MT19937(0))

    def process(indices, gender):
        nonlocal joint_weights, vertex_weights
        m_fnames = labels['fnames'][indices]
        m_shapes = labels['shapes'][indices]
        m_poses = labels['poses'][indices]
        m_cam_trans = labels['cam_trans'][indices]
        m_pred_verts = preds['vertices'][indices]
        m_pred_joints = preds['joints'][indices]
        m_pred_verts_subset = m_pred_verts[:, vertex_subset]

        m_vertex_weights = vertex_weights[indices]
        m_vertex_weights_subset = m_vertex_weights[:, vertex_subset]
        m_joint_weights = joint_weights[indices]


        if FLAGS.group_sizes is not None:
            groups = spu.groupby(range(len(m_fnames)), lambda i: m_fnames[i].split('_frame_')[0])
            #groups = spu.groupby(range(len(m_fnames)), lambda i: get_video_person(m_fnames[i]))
            m_fits_per_group = []
            all_indices = []

            for group_indices in groups.values():
                rng.shuffle(group_indices)
                all_indices.extend(group_indices)
                index_chunks = more_itertools.chunked(group_indices, FLAGS.group_sizes)
                for chunk_indices in index_chunks:
                    m_vertex_weights_subset = get_weights(
                        preds['vertex_uncertainties'][indices][chunk_indices],
                        axis=(-2, -1))[:, vertex_subset]
                    m_joint_weights = get_weights(
                        preds['joint_uncertainties'][indices][chunk_indices], axis=(-2, -1))
                    fit = fit_fn_scale(
                        m_pred_verts_subset[chunk_indices], m_pred_joints[chunk_indices],
                        m_vertex_weights_subset, m_joint_weights)
                    scale = fit['scale_corr'][..., np.newaxis, np.newaxis]
                    scale = scale / np.mean(scale)
                    fit = fit_fn(
                        m_pred_verts_subset[chunk_indices]/scale, m_pred_joints[chunk_indices]/scale,
                        m_vertex_weights_subset, m_joint_weights)
                    m_fits_per_group.append(fit)
            reverse_indices = np.argsort(all_indices)
            m_fits = {k: np.concatenate([f[k] for f in m_fits_per_group], axis=0)[reverse_indices]
                      for k in m_fits_per_group[0].keys()}
        else:
            m_fits = fit_fn(
                m_pred_verts_subset, m_pred_joints, m_vertex_weights_subset, m_joint_weights)

        # if gender == 'm':
        #     rep = 32
        #     a = tf.repeat(m_pred_verts_subset[:64], rep, axis=0)
        #     b = tf.repeat(m_pred_joints[:64], rep, axis=0)
        #     c = tf.repeat(m_vertex_weights_subset[:64], rep, axis=0)
        #     d = tf.repeat(m_joint_weights[:64], rep, axis=0)
        #
        #     timings = timeit.repeat(lambda: fit_fn(a, b, c, d), number=30, repeat=3)
        #     print(len(indices))
        #     timing = 1/(np.min(np.array(timings)) / 30 / (64*rep))
        #     print('Time', timing)

        # m_points = tf.concat([m_pred_verts, m_pred_joints], axis=1)
        # m_mean_points = tf.reduce_mean(m_points, axis=1)
        # print(list(m_fits['scale_corr'].numpy()))
        # m_fits['trans'] += m_mean_points * (m_fits['scale_corr'][:, tf.newaxis]-1)

        m_fit_res = bm(m_fits['pose_rotvecs'], m_fits['shape_betas'],
                       m_fits['trans'])  # /m_fits['scale_corr'][..., tf.newaxis])
        m_fit_verts = m_fit_res['vertices']
        m_fit_joints = m_fit_res['joints']
        m_fit_shapes = m_fits['shape_betas']

        m_ious_np, m_ious_fit, m_mve_np, m_mve_fit = compute_metrics(
            m_fnames, m_shapes, m_poses, m_cam_trans, m_pred_verts, m_fit_verts, m_pred_joints,
            m_fit_joints, gender, camera)

        m_pve_t_sc, m_pve_t = compute_pve_neutral_pose_scale_corrected(
            m_fit_shapes, m_shapes, gender)

        return m_ious_np, m_ious_fit, m_mve_np, m_mve_fit, m_pve_t_sc, m_pve_t

    i_males = np.argwhere([g[0] == 'm' for g in labels['genders']]).squeeze(-1)
    i_females = np.argwhere([g[0] == 'f' for g in labels['genders']]).squeeze(-1)

    m_ious_np, m_ious_fit, m_mve_np, m_mve_fit, m_pve_t_sc, m_pve_t = process(i_males, 'm')
    f_ious_np, f_ious_fit, f_mve_np, f_mve_fit, f_pve_t_sc, f_pve_t = process(i_females, 'f')

    mean_iou_np = np.mean(np.concatenate([m_ious_np, f_ious_np]))
    mean_iou_fit = np.mean(np.concatenate([m_ious_fit, f_ious_fit]))
    mve_np = np.mean(np.concatenate([m_mve_np, f_mve_np]))
    mve_fit = np.mean(np.concatenate([m_mve_fit, f_mve_fit]))

    pve_t_sc_all = np.concatenate([m_pve_t_sc, f_pve_t_sc])
    pve_t_all = np.concatenate([m_pve_t, f_pve_t])

    print('mIoU np', mean_iou_np)
    print('mIoU fit', mean_iou_fit)
    print('PVE-T-SC', np.mean(pve_t_sc_all))
    print('PVE-T', np.mean(pve_t_all))
    print('MVE np', mve_np)
    print('MVE fit', mve_fit)


def compute_metrics(fnames, shapes, poses, cam_trans, verts_pred, verts_fit, joints_pred,
                    joints_fit, gender, camera):
    bm = smpl.numpy.get_cached_body_model('smpl', gender)
    gt_res = bm(shape_betas=shapes, pose_rotvecs=poses)
    verts_gt = gt_res['vertices'] + cam_trans[:, np.newaxis]
    joints_gt = gt_res['joints'] + cam_trans[:, np.newaxis]

    mve_np = np.mean(
        np.linalg.norm((verts_gt - joints_gt[:, :1]) - (verts_pred - joints_pred[:, :1]), axis=-1),
        axis=-1)
    mve_fit = np.mean(
        np.linalg.norm((verts_gt - joints_gt[:, :1]) - (verts_fit - joints_fit[:, :1]), axis=-1),
        axis=-1)

    gt_masks = meshes_to_masks(verts_gt, bm.faces, camera, (512, 512))
    pred_masks = meshes_to_masks(verts_pred, bm.faces, camera, (512, 512))
    fit_masks = meshes_to_masks(verts_fit, bm.faces, camera, (512, 512))

    ious_np = np.array([rlemasklib.iou([p, gt]) for p, gt in zip(pred_masks, gt_masks)])
    ious_fit = np.array([rlemasklib.iou([p, gt]) for p, gt in zip(fit_masks, gt_masks)])
    return ious_np, ious_fit, mve_np, mve_fit  # np.zeros_like(ious_fit)


def meshes_to_masks(vertices, faces, camera, imshape):
    im, depth = smpl.render.render(vertices, faces, camera, imshape=imshape)
    masks = np.uint8(depth > 0)
    return [rlemasklib.encode(m) for m in masks]


def compute_pve_neutral_pose_scale_corrected(predicted_smpl_shape, target_smpl_shape, gender):
    smpl_gendered = smpl.numpy.get_cached_body_model('smpl', gender)
    smpl_neutral = smpl.numpy.get_cached_body_model('smpl', 'neutral')

    target_smpl_neutral_pose_output = smpl_gendered(shape_betas=target_smpl_shape)
    pred_smpl_neutral_pose_output = smpl_neutral(shape_betas=predicted_smpl_shape)

    pred_smpl_neutral_pose_vertices = pred_smpl_neutral_pose_output['vertices']
    target_smpl_neutral_pose_vertices = target_smpl_neutral_pose_output['vertices']

    # Rescale such that RMSD of predicted vertex mesh is the same as RMSD of target mesh.
    # This is done to combat scale vs camera depth ambiguity.
    pred_smpl_neutral_pose_vertices_rescale = scale_and_translation_transform_batch(
        pred_smpl_neutral_pose_vertices, target_smpl_neutral_pose_vertices)

    pred_smpl_neutral_pose_vertices_trans = scale_and_translation_transform_batch(
        pred_smpl_neutral_pose_vertices, target_smpl_neutral_pose_vertices, scale=False)

    # Compute PVE-T-SC
    pve_neutral_pose_scale_corrected = np.linalg.norm(
        pred_smpl_neutral_pose_vertices_rescale - target_smpl_neutral_pose_vertices,
        axis=-1)
    pve_neutral_pose = np.linalg.norm(
        pred_smpl_neutral_pose_vertices_trans - target_smpl_neutral_pose_vertices,
        axis=-1)
    return pve_neutral_pose_scale_corrected, pve_neutral_pose


def scale_and_translation_transform_batch(P, T, scale=True):
    """
    First normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    scale_factor = T_scale / P_scale if scale else 1

    P_transformed = P_trans * scale_factor + T_mean

    return P_transformed


if __name__ == '__main__':
    main()
