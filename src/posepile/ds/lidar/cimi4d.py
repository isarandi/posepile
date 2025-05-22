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

import numpy as np


@spu.picklecache(f'cimi4d.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    DATASET_ROOT = f'{DATA_ROOT}/lidar/CIMI4D'
    seq_names = sorted(os.listdir(f'{DATASET_ROOT}/XMU_V2'))
    examples = []

    with spu.ThrottledPool() as pool:
        for seq_name in seq_names:
            seq_dir = f'{DATASET_ROOT}/XMU_V2/{seq_name}'
            data = spu.load_pickle(glob.glob(f'{seq_dir}/*.pkl')[0])
            camera = load_camera(seq_dir)
            frame_paths = spu.natural_sorted(glob.glob(f'{seq_dir}/*_img/*.jpg'))

            for i_frame, frame_path in enumerate(spu.progressbar(frame_paths, desc=seq_name)):
                joints_cam = cam.world_to_camera(joints * 1000)
                if sampler.should_skip(joints_cam):
                    continue

                if not box:
                    continue

                bbox = np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]], np.float32)
                joints2d = cam.world_to_image(joints*1000)
                if not (
                        np.count_nonzero(boxlib.contains(bbox, joints2d)) >=6 and
                        np.all(joints_cam[:, 2] > 0)):
                    continue

                parameters = dict(type='smpl', gender=gender, pose=pose, shape=betas, trans=trans)
                ex = ds3d.Pose3DExample(
                    image_path=frame, camera=cam, bbox=bbox, mask=None, world_coords=None,
                    parameters=parameters)
                new_image_relpath = f'lidar_downscaled/CIMI4D/{seq_name}/{basename}'
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath), callback=examples.append,
                    kwargs=dict(extreme_perspective=True))

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples)


def load_camera(dirpath):
    content = spu.read_lines(f'{dirpath}/LiDAR2Cam_e.txt')
    lidar2cam_str = content.split('intrinsic_matrix')[0].strip()
    lidar2cam = np.array([list(map(float, line.split())) for line in lidar2cam_str.split('\n')])
    intrinsic_str = re.search(r'intrinsic_matrix = np.array\((.*?)\)', content, re.DOTALL).group(1)
    intrinsic_matrix = np.fromstring(intrinsic_str, sep=',').reshape(3, 3)
    distortion_str = re.search(r'distortion_coefficients = np.array\((.*?)\)', content, re.DOTALL).group(1)
    distortion_coefficients = np.fromstring(distortion_str, sep=',')

    world2lidar = np.linalg.inv(np.loadtxt(f'{dirpath}/lidar_p.txt'))
    world2lidar[:3, 3] /= 1000

    return cameralib.Camera(
        intrinsic_matrix=intrinsic_matrix,
        distortion_coeffs=distortion_coefficients,
        extrinsic_matrix=lidar2cam @ world2lidar,
        world_up=(0, 0, 1))

def select_items_by_index(iterable, indices):
    it = iter(enumerate(iterable))
    for i_wanted in indices:
        while True:
            i, item = next(it)
            if i == i_wanted:
                yield item
                break


if __name__ == '__main__':
    make_sloper4d()
