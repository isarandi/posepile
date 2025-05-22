import os.path as osp
import sys

import barecat
import cameravision
import more_itertools
import numpy as np
import posepile.joint_info
import simplepyutils as spu
import smpl
import smpl.tensorflow.full_fitting
import tensorflow as tf
from posepile.ds.agora.add_parametric import example_to_dict
from posepile.ds.spec.main import calibrate_extrinsics
from posepile.joint_info import JointInfo
from posepile.paths import CACHE_DIR, DATA_ROOT

sys.modules['data.joint_info'] = posepile.joint_info
sys.modules['data'] = posepile


def main():
    ds = spu.load_pickle(f'{CACHE_DIR}/spec2.pkl')
    root = f'{DATA_ROOT}/spec/spec-syn'
    anno_train = np.load(f'{root}/annotations/train.npz')
    anno_test = np.load(f'{root}/annotations/test.npz')

    body_model = smpl.tensorflow.SMPL(model_root='/work/sarandi/data/body_models/smpl', gender='n')
    body_model_forward = tf.function(body_model.__call__)

    body_model_np = smpl.numpy.get_cached_body_model()
    J_regressor_post_lbs = np.load(f'/work/sarandi/smpl_neutral_J_regressor_post_lbs.npy')
    fitter = smpl.tensorflow.full_fitting.Fitter(
        body_model, J_regressor_post_lbs, num_betas=10, l2_regularizer=5e-6, enable_kid=False)
    fit_fn = tf.function(fitter.fit)

    joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe'
    joint_info = JointInfo(joint_names, edges)

    openpose_joint_names = (
        'head,neck,rsho,relb,rwri,lsho,lelb,lwri,pelv,rhip,rkne,rank,lhip,lkne,lank,reye,leye,'
        'rear,lear,lto2,lto3,lhee,rto2,rto3,rhee').split(',')
    commons = 'neck,rsho,relb,rwri,lsho,lelb,lwri,pelv,rhip,rkne,rank,lhip,lkne,lank'.split(',')
    # S_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    i_3d = [joint_info.ids[n] for n in commons]
    i_2d = [openpose_joint_names.index(n) for n in commons]
    # S_selector = [joint_info.ids[n] for n in S_names]

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/spec_smpl.barecat',
            overwrite=True, readonly=False, auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        for phase_name, phase, a in zip('train test'.split(), [0, 2], [anno_train, anno_test]):
            examples = ds.examples[phase]

            a_pose = np.array(a['pose'])
            a_shape = np.array(a['shape'])
            a_cam_rotmat = np.array(a['cam_rotmat'])
            a_cam_int = np.array(a['cam_int'])
            a_openpose_gt = np.array(a['openpose_gt'])

            for exs in more_itertools.chunked(spu.progressbar(examples), 256):
                i_perss = [
                    int(osp.splitext(osp.basename(ex.image_path))[0].split('_')[-1])
                    for ex in exs]
                pose_unk_res = body_model_np(
                    np.array(a_pose[i_perss]), np.array(a_shape[i_perss]))
                pose_unks = pose_unk_res['joints'] * 1000
                verts_unks = pose_unk_res['vertices'] * 1000

                world_vertss = []
                for ex, pose_unk, verts_unk, i_pers in zip(exs, pose_unks, verts_unks, i_perss):
                    openpose_gt = a_openpose_gt[i_pers]
                    cam_rotmat = a_cam_rotmat[i_pers]
                    cam_int = a_cam_int[i_pers]

                    pose2d = openpose_gt[..., :2]
                    cam = cameravision.Camera(
                        rot_world_to_cam=cam_rotmat, intrinsic_matrix=cam_int, world_up=(0, -1, 0))
                    rot, trans = calibrate_extrinsics(pose2d[i_2d], pose_unk[i_3d], cam_int)
                    scale_factor = 1  # get_scale(S) / get_scale(pose_unk[S_selector])
                    camcoords = (pose_unk @ rot.T + trans) * scale_factor
                    world_coords = cam.camera_to_world(camcoords)

                    camcoords = (verts_unk @ rot.T + trans) * scale_factor
                    world_verts = cam.camera_to_world(camcoords)
                    world_vertss.append(world_verts)

                res = fit_fn(np.array(world_vertss) / 1000, n_iter=8)
                for ex, pose, shape, trans in zip(
                        exs, res['pose_rotvecs'].numpy(), res['shape_betas'].numpy(),
                        res['trans'].numpy()):
                    ex.parameters = dict(
                        type='smpl', gender='neutral', pose=pose, shape=shape,
                        kid_factor=np.float32(0), trans=trans)
                    bc_writer[
                        f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


if __name__ == '__main__':
    main()
