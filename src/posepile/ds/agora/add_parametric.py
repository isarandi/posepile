import functools
import glob
import os.path as osp
import sys

import barecat
import cv2
import numpy as np
import posepile.joint_info
import rlemasklib
import simplepyutils as spu
import smpl.numpy
import transforms3d
from posepile.paths import DATA_ROOT

sys.modules['data.joint_info'] = posepile.joint_info
sys.modules['data'] = posepile


def main():
    agora_root = '/globalwork/datasets/AGORA/'
    dfs = [spu.load_pickle(p) for p in glob.glob(f'{agora_root}/Cam/*.pkl')]

    impath_to_row = {}
    for df in dfs:
        for i, path in enumerate(df['imgPath']):
            impath_to_row[path] = df.iloc[i]

    ds = spu.load_pickle('/work/sarandi/data/agora2.pkl')

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/agora_smplx.barecat', overwrite=True, readonly=False,
            auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        for phase_name, phase in zip('train val test'.split(), [0, 1, 2]):
            examples = ds.examples[phase]
            for i, ex in enumerate(spu.progressbar(examples)):
                orig_image_path, person_id = parse_cropped_imagepath(ex.image_path)
                input_row = impath_to_row[orig_image_path]
                smplx_path = input_row['gt_path_smplx'][person_id]
                is_kid = input_row['kid'][person_id]
                gender = input_row['gender'][person_id]
                if is_kid and gender == 'female':
                    gender = 'neutral'

                pose, betas, transl, kid_factor = load_smplx_params(
                    spu.replace_extension(f'{agora_root}/{smplx_path}', '.pkl'),
                    is_kid=is_kid, gender=gender)
                pose, transl = adjust_body(
                    input_row, person_id, pose, betas, transl, kid_factor, gender,
                    model_name='smplx')

                ex.parameters = dict(
                    type='smplx', gender=gender, pose=pose,
                    shape=betas, kid_factor=kid_factor,
                    trans=transl)

                bc_writer[
                    f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


def example_to_dict(ex):
    parameters = {**ex.parameters}
    if parameters.get('kid_factor', 0) == 0:
        del parameters['kid_factor']

    result = dict(
        impath=ex.image_path,
        bbox=np.round(ex.bbox).astype(np.int16),
        parameters=ex.parameters,
        cam=dict(
            rotvec_w2c=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            loc=ex.camera.t,
            intr=ex.camera.intrinsic_matrix[:2],
            up=ex.camera.world_up
        )
    )
    if (ex.camera.distortion_coeffs is not None and
            np.count_nonzero(ex.camera.distortion_coeffs) > 0):
        result['cam']['distcoef'] = ex.camera.distortion_coeffs

    if ex.mask is not None:
        result['mask'] = rlemasklib.compress(ex.mask, zlevel=-1)
    return result


def parse_cropped_imagepath(p):
    filename = osp.basename(p)
    noext = osp.splitext(filename)[0]
    parts = noext.split('_')
    firstpart = '_'.join(parts[:-1])
    person_id = int(parts[-1])
    return firstpart + '.png', person_id


def load_smpl_params(path, is_kid):
    import torch
    with torch.device('cpu'):
        data = spu.load_pickle(path)
        get = lambda x: data[x].detach().cpu().numpy()[0].astype(np.float32)
        pose = np.concatenate([get('root_pose'), get('body_pose')], axis=0).reshape(-1)
        betas = get('betas')
        trans = get('translation')
        if is_kid:
            kid_factor = betas[-1]
            betas = betas[:-1]
        else:
            kid_factor = np.float32(0)
        return pose, betas, trans, kid_factor, 'neutral'


def load_smplx_params(path, is_kid, gender):
    import torch
    with torch.device('cpu'):
        data = spu.load_pickle(path)
        get = lambda x: data[x][0].astype(np.float32)

        pose_parts = [
            'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
        ]
        pose = np.concatenate([get(x) for x in pose_parts], axis=0).reshape(-1)
        left_hand_mean, right_hand_mean = get_hand_means(gender)
        left_hand_pose = get('left_hand_pose') + left_hand_mean
        right_hand_pose = get('right_hand_pose') + right_hand_mean
        pose = np.concatenate([pose, left_hand_pose, right_hand_pose], axis=0)
        betas = get('betas')
        if is_kid:
            kid_factor = betas[-1]
            betas = betas[:-1]
        else:
            kid_factor = np.float32(0)

        betas = np.concatenate([betas, get('expression')], axis=0).reshape(-1)
        trans = get('transl')

        return pose, betas, trans, kid_factor


def adjust_body(input_row, person_id, pose, betas, transl, kid_factor, gender, model_name='smpl'):
    # AGORA stores a person yaw angle and location in addition to a rotation already
    # encoded into the SMPL pose params and a translation in the SMPL params, too.
    # Here we merge all this info into the SMPL parameters, to make things simpler.

    yaw = input_row['Yaw'][person_id]
    # R_yaw = transforms3d.euler.euler2mat(-np.deg2rad(yaw), 0, 0, 'syxz')
    # rot_change = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], np.float32) @ R_yaw
    rot_change = transforms3d.euler.euler2mat(-np.deg2rad(yaw + 90), 0, np.pi, 'syxz')
    globally_rotate_inplace(pose, rot_change)
    X = input_row['X'][person_id]
    Y = input_row['Y'][person_id]
    Z = input_row['Z'][person_id]
    person_location = np.array([X, Y, Z]) / 100
    offset = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], np.float32) @ person_location

    body_model = smpl.numpy.get_cached_body_model(model_name=model_name, gender=gender)
    pelvis = (
            body_model.J_template[0] +
            body_model.J_shapedirs[0, :, :betas.shape[0]] @ betas +
            body_model.kid_J_shapedir[0] * kid_factor +
            transl)
    transl += pelvis @ (rot_change - np.eye(3)) + offset
    return pose, transl


def globally_rotate_inplace(pose_rotvec, rot_change):
    current_rotmat = cv2.Rodrigues(pose_rotvec[:3])[0]
    new_rotmat = rot_change @ current_rotmat
    pose_rotvec[:3] = cv2.Rodrigues(new_rotmat)[0][:, 0]


@functools.lru_cache()
def get_hand_means(gender):
    gender_map = dict(f='FEMALE', m='MALE', n='NEUTRAL')
    a = np.load(f'{DATA_ROOT}/body_models/smplx/SMPLX_{gender_map[gender[0].lower()]}.npz')
    return a['hands_meanl'], a['hands_meanr']


if __name__ == '__main__':
    main()
