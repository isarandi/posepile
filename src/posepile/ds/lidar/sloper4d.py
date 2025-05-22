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


@spu.picklecache(f'sloper4d.pkl', min_time="2023-12-01T13:35:22")
def make_sloper4d():
    seq_dirs = spu.sorted_recursive_glob(f'{DATA_ROOT}/lidar/SLOPER4D/seq*')
    examples = []

    with spu.ThrottledPool() as pool:
        for seq_dir in seq_dirs:
            seq_name = osp.basename(seq_dir)
            ds_params = spu.load_json(f'{seq_dir}/dataset_params.json')
            frame_rate = ds_params['RGB_info']['framerate']
            d = spu.load_pickle(f'{seq_dir}/{seq_name}_labels.pkl')
            cameras_seq = get_cameras(d)
            betas_seq = np.array(d['RGB_frames']['beta'])
            trans_seq = np.array(d['RGB_frames']['global_trans'])
            poses_seq = np.array(d['RGB_frames']['smpl_pose'], np.float32)
            boxes_seq = d['RGB_frames']['bbox']

            gender = d['SMPL_info']['gender']
            joints_seq = smpl.numpy.get_cached_body_model('smpl', gender)(
                poses_seq, betas_seq, trans_seq, return_vertices=False)['joints']

            sampler = AdaptivePoseSampler2(
                100, True, True, 100)
            video_reader = imageio.get_reader(f'{seq_dir}/rgb_data/{seq_name}.MP4')

            basenames = d['RGB_frames']['file_basename']
            i_frames = [int(round(float(osp.splitext(p)[0]) * frame_rate)) for p in basenames]

            # iterate over the frames that correspond to the indices of i_frames
            frames = select_items_by_index(video_reader, i_frames)

            for frame, cam, joints, pose, betas, trans, box, basename in zip(
                    frames, spu.progressbar(cameras_seq), joints_seq, poses_seq, betas_seq,
                    trans_seq, boxes_seq, basenames):

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
                new_image_relpath = f'lidar_downscaled/SLOPER4D/{seq_name}/{basename}'
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath), callback=examples.append,
                    kwargs=dict(extreme_perspective=True))

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples)


def get_cameras(data):
    fx, fy, cx, cy = data['RGB_info']['intrinsics']
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    extrinsics = data['RGB_frames']['cam_pose']
    distortion_coeffs = np.array(data['RGB_info']['dist'], np.float32)
    result = [cameralib.Camera(
        intrinsic_matrix=intrinsic_matrix.copy(),
        distortion_coeffs=distortion_coeffs.copy(),
        extrinsic_matrix=e, world_up=(0, 0, 1)) for e in extrinsics]
    for c in result:
        c.t *= 1000
    return result


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
