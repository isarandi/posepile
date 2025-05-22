import argparse
import os.path as osp

import numpy as np
import simplepyutils as spu
import smpl.numpy
import smpl.tensorflow
import trimesh
from posepile.ds.tdpw.eval import to_latex
from posepile.paths import DATA_ROOT
from posepile.util.rigid_alignment import rigid_align_many
from simplepyutils import FLAGS
import cv2
import cameravision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, required=True)
    parser.add_argument('--fitted', action=spu.argparse.BoolAction)
    parser.add_argument('--camspace', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)

    gt_meshes = get_gt_meshes()
    pred_meshes =  get_pred_meshes_fitted() if FLAGS.fitted else get_pred_meshes()

    hand_dict = spu.load_pickle(f'{DATA_ROOT}/body_models/smplx/MANO_SMPLX_vertex_ids.pkl')
    lhand_indices = hand_dict['left_hand']
    rhand_indices = hand_dict['right_hand']
    face_indices = np.load(f'{DATA_ROOT}/body_models/smplx/SMPL-X__FLAME_vertex_ids.npy')
    body_indices = np.setdiff1d(
        np.arange(10475), np.concatenate([lhand_indices, rhand_indices, face_indices]))

    bm = smpl.numpy.get_cached_body_model('smplx', 'neutral')

    # MPVPE: all, l_hand, r_hand, hand, face
    i_pelvis = 0
    i_lwrist = 20
    i_rwrist = 21
    i_neck = 12
    gt_joints24 = bm.J_regressor @ gt_meshes
    pred_joints24 = bm.J_regressor @ pred_meshes

    mpvpe_all = get_mpvpe(
        pred_meshes, gt_meshes, pred_joints24, gt_joints24, np.arange(10475), i_pelvis)
    mpvpe_lhand = get_mpvpe(
        pred_meshes, gt_meshes, pred_joints24, gt_joints24, lhand_indices, i_lwrist)
    mpvpe_rhand = get_mpvpe(
        pred_meshes, gt_meshes, pred_joints24, gt_joints24, rhand_indices, i_rwrist)
    mpvpe_hand = (mpvpe_lhand + mpvpe_rhand) / 2
    mpvpe_face = get_mpvpe(
        pred_meshes, gt_meshes, pred_joints24, gt_joints24, face_indices, i_neck)

    # PA-MPVPE: all, body, l_hand, r_hand, hand, face
    pa_mpvpe_all = get_pa_mpvpe(pred_meshes, gt_meshes, np.arange(10475))
    pa_mpvpe_body = get_pa_mpvpe(pred_meshes, gt_meshes, body_indices)
    pa_mpvpe_lhand = get_pa_mpvpe(pred_meshes, gt_meshes, lhand_indices)
    pa_mpvpe_rhand = get_pa_mpvpe(pred_meshes, gt_meshes, rhand_indices)
    pa_mpvpe_hand = (pa_mpvpe_lhand + pa_mpvpe_rhand) / 2
    pa_mpvpe_face = get_pa_mpvpe(pred_meshes, gt_meshes, face_indices)

    # PA-MPJPE: body, l_hand, r_hand, hand, face
    J_regressor14 = spu.load_pickle(f'{DATA_ROOT}/body_models/smplx/SMPLX_to_J14.pkl')
    lhand_reg, rhand_reg = make_hand_regressors()

    pa_mpjpe_body = get_pa_mpjpe(pred_meshes, gt_meshes, J_regressor14)
    pa_mpjpe_lhand = get_pa_mpjpe(pred_meshes, gt_meshes, lhand_reg)
    pa_mpjpe_rhand = get_pa_mpjpe(pred_meshes, gt_meshes, rhand_reg)
    pa_mpjpe_hand = (pa_mpjpe_lhand + pa_mpjpe_rhand) / 2

    metrics = [
        mpvpe_all, mpvpe_hand, mpvpe_face, pa_mpvpe_all, pa_mpvpe_body, pa_mpvpe_hand,
        pa_mpvpe_face, pa_mpjpe_body, pa_mpjpe_hand]
    metric_names = [
        'MPVPE_all', 'MPVPE_hand', 'MPVPE_face', 'PA-MPVPE_all', 'PA-MPVPE_body', 'PA-MPVPE_hand',
        'PA-MPVPE_face', 'PA-MPJPE_body', 'PA-MPJPE_hand']
    print(' & '.join(metric_names))
    print(to_latex(metrics))


def get_mpvpe(pred_meshes, gt_meshes, pred_joints, gt_joints, i_verts, i_joint_ref):
    shift = gt_joints[:, i_joint_ref] - pred_joints[:, i_joint_ref]
    pred_shifted = pred_meshes[:, i_verts] + shift[:, np.newaxis]
    return np.mean(np.linalg.norm(pred_shifted - gt_meshes[:, i_verts], axis=-1)) * 1000


def get_pa_mpvpe(pred_meshes, gt_meshes, i_verts):
    pred_aligned = rigid_align_many(pred_meshes[:, i_verts], gt_meshes[:, i_verts])
    return np.mean(np.linalg.norm(pred_aligned - gt_meshes[:, i_verts], axis=-1)) * 1000


def get_pa_mpjpe(pred_meshes, gt_meshes, regressor):
    pred_joints = regressor @ pred_meshes
    gt_joints = regressor @ gt_meshes
    pred_aligned = rigid_align_many(pred_joints, gt_joints)
    return np.mean(np.linalg.norm(pred_aligned - gt_joints, axis=-1)) * 1000


def make_hand_regressors():
    bm = smpl.numpy.get_cached_body_model('smplx', 'neutral')
    n_verts = bm.num_vertices
    n_hand_joints = 21
    regressor = bm.J_regressor
    lhand_regressor = np.zeros((n_hand_joints, n_verts), np.float32)
    rhand_regressor = np.zeros((n_hand_joints, n_verts), np.float32)
    lhand_regressor[:4] = regressor[[20, 37, 38, 39]]
    lhand_regressor[4, 5361] = 1
    lhand_regressor[5:8] = regressor[[25, 26, 27]]
    lhand_regressor[8, 4933] = 1
    lhand_regressor[9:12] = regressor[[28, 29, 30]]
    lhand_regressor[12, 5058] = 1
    lhand_regressor[13:16] = regressor[[34, 35, 36]]
    lhand_regressor[16, 5169] = 1
    lhand_regressor[17:20] = regressor[[31, 32, 33]]
    lhand_regressor[20, 5286] = 1

    rhand_regressor[:4] = regressor[[21, 52, 53, 54]]
    rhand_regressor[4, 8079] = 1
    rhand_regressor[5:8] = regressor[[40, 41, 42]]
    rhand_regressor[8, 7669] = 1
    rhand_regressor[9:12] = regressor[[43, 44, 45]]
    rhand_regressor[12, 7794] = 1
    rhand_regressor[13:16] = regressor[[49, 50, 51]]
    rhand_regressor[16, 7905] = 1
    rhand_regressor[17:20] = regressor[[46, 47, 48]]
    rhand_regressor[20, 8022] = 1
    return lhand_regressor, rhand_regressor

def get_pred_meshes_fitted():
    bm = smpl.numpy.get_cached_body_model('smplx', 'neutral')
    data = np.load(FLAGS.pred_path)
    res = bm(data['pose'], data['betas'], data['trans'])
    return res['vertices']


def get_pred_meshes():
    data = spu.load_pickle(FLAGS.pred_path)
    return np.stack(
        [data[f'{i + 1:02d}_img.png']['vertices'] /1000 for i in range(100)]).astype(np.float32)


def get_gt_meshes():
    dirpath = osp.join(DATA_ROOT, 'ehf')
    ply_names = [f'{i + 1:02d}_align.ply' for i in range(100)]
    ply_paths = [osp.join(dirpath, name) for name in ply_names]
    meshes = [trimesh.load(ply_path) for ply_path in ply_paths]
    verts = np.stack([np.array(m.vertices).astype(np.float32) for m in meshes])

    if FLAGS.camspace:
        camera = get_camera()
        verts = camera.world_to_camera(verts)

    return verts



def get_camera():
    f = 1498.22426237
    cx = 790.263706
    cy = 578.90334
    R = cv2.Rodrigues(np.array([-2.98747896, 0.01172457, -0.05704687], np.float32))[0]
    return cameravision.Camera(
        intrinsic_matrix=np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32),
        rot_world_to_cam=R,
        trans_after_rot=np.array([-0.03609917, 0.43416458, 2.37101226], np.float32),
        world_up=(0, 1, 0))

if __name__ == "__main__":
    main()
