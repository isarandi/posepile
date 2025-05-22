import functools
import re
import sys

import barecat
import numpy as np
import posepile.joint_info
import simplepyutils as spu
from posepile.ds.agora.add_parametric import example_to_dict
from posepile.paths import CACHE_DIR, DATA_ROOT

sys.modules['data.joint_info'] = posepile.joint_info
sys.modules['data'] = posepile


def main():
    rich_root = f'{DATA_ROOT}/rich'
    ds = spu.load_pickle(f'{CACHE_DIR}/rich.pkl')
    males = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 20]

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/rich_smplx.barecat',
            overwrite=True, readonly=False, auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        for phase_name, phase in zip('train val test'.split(), [0, 1, 2]):
            examples = ds.examples[phase]
            for i, ex in enumerate(spu.progressbar(examples)):
                phase_name, seq_name, cam_id, i_frame, subj = parse_imagepath(ex.image_path)
                gender = 'male' if subj in males else 'female'
                pose, betas, trans, kid_factor, expression = load_smplx_params(
                    f'{rich_root}/{phase_name}_body/{seq_name}/{i_frame:05d}/{subj:03d}.pkl',
                    gender)

                ex.parameters = dict(
                    type='smplx', gender=gender, pose=pose,
                    shape=betas, kid_factor=kid_factor, trans=trans, expression=expression)
                bc_writer[
                    f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


def parse_imagepath(p):
    m = re.match(
        r'rich_downscaled/(?P<phase_name>train|val)/'
        r'(?P<seq_name>.+?)/cam_(?P<cam>\d+)/(?P<frame>\d+)_(\d+)_(?P<subj>\d+).jpg',
        p)
    phase_name = m['phase_name']
    seq_name = m['seq_name']
    cam_id = int(m['cam'])
    i_frame = int(m['frame'])
    subj = int(m['subj'])
    return phase_name, seq_name, cam_id, i_frame, subj


def load_smplx_params(src, gender=None):
    if isinstance(src, str):
        data = spu.load_pickle(src)
    else:
        data = src
    if gender is None:
        gender = data['gender']
    get = lambda x: data[x][0].astype(np.float32)
    pose_parts = [
        'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
    ]
    pose = np.concatenate([get(x).reshape(-1) for x in pose_parts], axis=0)
    left_hand_mean, right_hand_mean, left_hand_mat, right_hand_mat = get_hand_pca(gender)
    left_hand_components = get('left_hand_pose')
    right_hand_components = get('right_hand_pose')
    left_hand_pose = (
            left_hand_components @ left_hand_mat[:left_hand_components.shape[-1]] +
            left_hand_mean)
    right_hand_pose = (
            right_hand_components @ right_hand_mat[:right_hand_components.shape[-1]] +
            right_hand_mean)
    pose = np.concatenate([pose, left_hand_pose, right_hand_pose], axis=0)
    # betas = np.concatenate([get('betas'), get('expression')], axis=0).reshape(-1)
    kid_factor = np.float32(0)
    return pose, get('betas'), get('transl'), kid_factor, get('expression')


@functools.lru_cache()
def get_hand_pca(gender):
    gender_map = dict(f='FEMALE', m='MALE', n='NEUTRAL')
    a = np.load(f'{DATA_ROOT}/body_models/smplx/SMPLX_{gender_map[gender[0].lower()]}.npz')
    return a['hands_meanl'], a['hands_meanr'], a['hands_componentsl'], a['hands_componentsr']


if __name__ == '__main__':
    main()
