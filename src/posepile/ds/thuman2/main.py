import glob
import os.path as osp

import boxlib
import numpy as np
import posepile.datasets3d as ds3d
import scipy.optimize
import simplepyutils as spu
from posepile.paths import DATA_ROOT
import functools

DATASET_NAME = 'thuman2'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    make_dataset()


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-20T03:01:47")
def make_dataset():
    examples = []
    camera_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/{DATASET_NAME}_render/*/cameras.pkl')

    for camera_path in spu.progressbar(camera_paths):
        seq_id = osp.basename(osp.dirname(camera_path))

        pose, shape, trans, expression, scale = load_smplx_params(
            f'{DATASET_DIR}/smplx/{seq_id}/smplx_param.pkl', gender='male')
        cameras = spu.load_pickle(camera_path)
        masks = spu.load_pickle(f'{DATA_ROOT}/{DATASET_NAME}_render/{seq_id}/masks.pkl')

        for i_gen, (camera, mask) in enumerate(zip(cameras, masks)):
            impath = f'{DATA_ROOT}/thuman2_render/{seq_id}/{i_gen:02d}.jpg'
            image_relpath = osp.relpath(impath, f'{DATA_ROOT}/{DATASET_NAME}_render')
            bbox = boxlib.bb_of_mask(mask)

            parameters = dict(
                type='smplx', gender='male', pose=pose,
                shape=shape, trans=trans, expression=expression)
            camera.t /= scale
            ex = ds3d.Pose3DExample(
                image_path=f'{DATASET_NAME}_render/{image_relpath}',
                camera=camera, bbox=bbox,
                parameters=parameters, world_coords=None, mask=mask)
            examples.append(ex)

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples)


def load_smplx_params(path, gender=None):
    data = spu.load_pickle(path)
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
    scale = get('scale')
    return (
        pose, get('betas'), np.array(data['translation'], np.float32) / scale, get('expression'),
        scale)


@functools.lru_cache()
def get_hand_pca(gender):
    gender_map = dict(f='FEMALE', m='MALE', n='NEUTRAL')
    a = np.load(f'{DATA_ROOT}/body_models/smplx/SMPLX_{gender_map[gender[0].lower()]}.npz')
    return a['hands_meanl'], a['hands_meanr'], a['hands_componentsl'], a['hands_componentsr']


if __name__ == '__main__':
    main()
