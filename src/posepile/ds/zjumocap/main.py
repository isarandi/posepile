import os
import os.path as osp

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import smpl.numpy
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from scipy.spatial.transform import Rotation as R

DATASET_NAME = 'zjumocap'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    bm = smpl.numpy.get_cached_body_model('smpl', 'neutral')
    seq_names = sorted(os.listdir(DATASET_DIR))
    examples = []

    with spu.ThrottledPool() as pool:
        for seq_name in spu.progressbar(seq_names):
            seq_dir = f'{DATASET_DIR}/{seq_name}'
            data = np.load(f'{seq_dir}/annots.npy', allow_pickle=True).item()

            first_impath = data['ims'][0]['ims'][0]
            world_up = (0, 0, -1) if 'Camera_B' in first_impath else (0, 0, 1)
            cameras = load_cameras(data, world_up)
            n_frames = len(data['ims'])

            sampler = AdaptivePoseSampler2(100, True, True, 100)
            first_index = 0 if osp.exists(f'{seq_dir}/new_params/0.npy') else 1
            for i_frame in spu.progressbar(range(n_frames), desc=seq_name):
                new_params = np.load(
                    f'{seq_dir}/new_params/{i_frame + first_index}.npy', allow_pickle=True).item()

                pose = new_params['poses'].squeeze(0)
                shape = new_params['shapes'].squeeze(0)
                trans = new_params['Th'].squeeze(0)
                rot = R.from_rotvec(new_params['Rh'].squeeze(0)).as_matrix()
                pose, trans = bm.rototranslate(rot, trans, pose, shape, np.zeros(3))
                params = dict(type='smpl', gender='neutral', pose=pose, shape=shape, trans=trans)

                joints = bm.single(pose, shape, trans, return_vertices=False)['joints'] * 1000
                if sampler.should_skip(joints):
                    continue

                for im_seqrelpath, cam in zip(data['ims'][i_frame]['ims'], cameras):
                    impath = f'{seq_dir}/{im_seqrelpath}'
                    mask_seqrelpath = 'mask/'+im_seqrelpath.replace('.jpg', '.png')
                    maskpath = f'{seq_dir}/{mask_seqrelpath}'
                    mask = np.uint8(imageio.imread(maskpath) > 0) * 255
                    bbox = boxlib.bb_of_mask(mask)
                    ex = ds3d.Pose3DExample(
                        image_path=impath, camera=cam, parameters=params, world_coords=None,
                        bbox=bbox, mask=mask)
                    new_image_relpath = f'zjumocap_downscaled/{seq_name}/{im_seqrelpath}'
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        callback=examples.append)

    return ds3d.Pose3DDataset(
        ds3d.JointInfo([], []), examples)


def load_cameras(data, world_up):
    K, R, T, D = [np.array(data['cams'][x]) for x in 'KRTD']
    return [cameralib.Camera(
        intrinsic_matrix=k, rot_world_to_cam=r, trans_after_rot=np.squeeze(t, -1),
        distortion_coeffs=np.squeeze(d, -1), world_up=world_up)
        for k, r, t, d in zip(K, R, T, D)]


if __name__ == '__main__':
    make_dataset()
