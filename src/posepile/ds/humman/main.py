import os.path as osp

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import smpl.numpy
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

DATASET_NAME = 'humman'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    examples = []
    seq_names = spu.read_lines(f'{DATASET_DIR}/train.txt')
    bm = smpl.numpy.get_cached_body_model('smpl', 'neutral')

    with spu.ThrottledPool() as pool:
        for seq_name in spu.progressbar(seq_names):
            seq_dir = f'{DATASET_DIR}/{seq_name}'
            cameras = load_cameras(f'{seq_dir}/cameras.json')
            param_paths = spu.sorted_recursive_glob(f'{seq_dir}/smpl_params/*.npz')
            frame_ids = [osp.splitext(osp.basename(p))[0] for p in param_paths]
            sampler = AdaptivePoseSampler2(
                100, True, True, 100)

            for frame_id in spu.progressbar(frame_ids, desc=seq_name, unit='frames', leave=False):
                smpl_data = np.load(f'{seq_dir}/smpl_params/{frame_id}.npz')
                pose = np.concatenate([smpl_data['global_orient'], smpl_data['body_pose']])
                shape = smpl_data['betas']
                trans = smpl_data['transl']
                parameters = dict(
                    type='smpl', gender='neutral', pose=pose, shape=shape, trans=trans)

                joints = bm.single(pose, shape, trans, return_vertices=False)['joints'] * 1000
                if sampler.should_skip(joints):
                    continue

                for i_cam, camera in enumerate(cameras):
                    mask = imageio.imread(
                        f'{seq_dir}/kinect_mask/kinect_{i_cam:03d}/{frame_id}.png')
                    bbox = boxlib.bb_of_mask(mask)
                    impath = f'{seq_dir}/kinect_color/kinect_{i_cam:03d}/{frame_id}.jpg'
                    ex = ds3d.Pose3DExample(
                        image_path=impath, camera=camera, bbox=bbox, mask=mask,
                        parameters=parameters, world_coords=None)

                    new_image_relpath = (
                        f'humman_downscaled/{seq_name}/kinect_color/'
                        f'kinect_{i_cam:03d}/{frame_id}.jpg')
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        kwargs=dict(downscale_input_for_antialias=True, downscale_at_decode=False),
                        callback=examples.append)

    return ds3d.Pose3DDataset(JointInfo([], []), examples)


def load_cameras(path):
    cam_data = spu.load_json(path)
    result = [
        cameralib.Camera(
            intrinsic_matrix=cam_data[f'kinect_color_{i:03d}']['K'],
            rot_world_to_cam=cam_data[f'kinect_color_{i:03d}']['R'],
            trans_after_rot=np.array(cam_data[f'kinect_color_{i:03d}']['T'], np.float32) * 1000,
            world_up=(0, -1, 0))
        for i in range(10)]

    world_up = np.sum([-c.R[1] for c in result], axis=0)
    world_up /= np.linalg.norm(world_up)
    for c in result:
        c.world_up = world_up
    return result


if __name__ == '__main__':
    make_dataset()
