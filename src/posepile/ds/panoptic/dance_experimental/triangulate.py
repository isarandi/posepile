import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
import argparse

import more_itertools
import numpy as np
import posepile.ds.panoptic.main as panoptic_main
import posepile.ds.experimental.cwi.triangulate as cwi_triangulate
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS
from posepile.joint_info import JointInfo
import simplepyutils as spu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--detector-flip-aug', action=spu.argparse.BoolAction)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--write-video', action=spu.argparse.BoolAction)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--camera', type=str, default='free')
    parser.add_argument('--viz', action=spu.argparse.BoolAction)
    parser.add_argument('--high-quality-viz', action=spu.argparse.BoolAction)
    parser.add_argument('--skeleton-types-file', type=str)
    parser.add_argument('--skeleton', type=str, default='smpl+head_30')
    spu.initialize(parser)
    ji3d = get_joint_info()

    # seq_names = [f'160317_moonbaby{i}' for i in range(1, 4)]
    seq_names = (
            [f'150821_dance{i}' for i in range(1, 6)] +
            [f'160317_moonbaby{i}' for i in range(1, 4)])

    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
        seq_names = seq_names[i_task:i_task + 1]

    root = f'{DATA_ROOT}/panoptic'

    for seq_name in seq_names:
        n_views = 29 if 'dance' in seq_name else 10
        output_path = f'{FLAGS.output_path}/{seq_name}_triang.pkl'
        if osp.exists(output_path):
            continue

        seq_dir = f'{root}/{seq_name}'

        if 'dance' in seq_name:
            camera_type = 'hd'
            camera_names = [f'00_{i:02d}' for i in range(0, 30) if i != 23]
            cameras = panoptic_main.get_cameras(
                f'{seq_dir}/calibration_{seq_name}_corrected_full.json', camera_names)
        else:
            camera_type = 'kinect'
            camera_names = [f'50_{i:02d}' for i in range(1, 11)]
            cameras = panoptic_main.get_cameras(
                f'{seq_dir}/calibration_{seq_name}.json', camera_names)

        cameras = [cameras[name] for name in camera_names]
        cameras = [c.undistort(inplace=False) for c in cameras]

        if 'dance' in seq_name:
            video_paths = [
                f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}_undistorted.mp4'
                for cam_name in camera_names]
        else:
            video_paths = [
                f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}_undistorted.mp4'
                for cam_name in camera_names]

        relpaths = [osp.splitext(osp.relpath(p, root))[0] for p in video_paths]
        pred_paths = [f'{FLAGS.output_path}/{r}.pkl' for r in relpaths]

        prediction_dicts = [spu.load_pickle(p) for p in pred_paths]
        n_frames = len(prediction_dicts[0]['boxes'])
        predictions = [zip(pd['boxes'], pd['poses3d']) for pd in prediction_dicts]
        if camera_type == 'kinect':
            indices = panoptic_main.get_kinect_sync_indices(seq_dir, seq_name, i_ref=0)
            predictions = [pick_indices(seq, i) for seq, i in zip(predictions, indices)]
        predictions = spu.roundrobin(predictions, [1] * n_views)

        triang_results = np.full(
            shape=[n_frames, ji3d.n_joints, 3], dtype=np.float32, fill_value=np.nan)
        index_selector = spu.load_pickle(FLAGS.skeleton_types_file)[FLAGS.skeleton]['indices']
        with spu.ThrottledPool(initializer=init_worker) as pool:
            for i_frame, preds in spu.progressbar(
                    enumerate(more_itertools.chunked(predictions, n_views)), total=n_frames):
                world_poses_per_cam = [poses[:, :, index_selector] for boxes, poses in preds]
                pool.apply_async(
                    triangulate, (cameras, world_poses_per_cam, ji3d),
                    callback=spu.itemsetter(triang_results, i_frame))

        spu.dump_pickle(triang_results, output_path)


def triangulate(cameras, world_poses_per_cam, ji3d):
    world_pose_per_cam = cwi_triangulate.find_main_person(
        world_poses_per_cam, ji3d, root_name='pelv_smpl', n_aug=5)
    campose_per_cam = np.array([
        c.world_to_camera(p) for c, p in zip(cameras, world_pose_per_cam)])
    is_valid = np.logical_not(np.any(np.isnan(campose_per_cam), axis=(1, 2, 3)))
    campose_per_cam_valid = campose_per_cam[is_valid]
    cameras_valid = [c for c, v in zip(cameras, is_valid) if v]
    campose_per_cam_valid = cwi_triangulate.mask_and_average(
        campose_per_cam_valid)

    if np.any(is_valid):
        triang = cwi_triangulate.triangulate_poses(
            cameras_valid, campose_per_cam_valid, imshape=(1080, 1920))
    else:
        triang = np.full(shape=[ji3d.n_joints, 3], dtype=np.float32, fill_value=np.nan)
    return triang


def init_worker():
    pass


import os.path as osp

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'


def get_joint_info():
    d = spu.load_pickle(FLAGS.skeleton_types_file)[FLAGS.skeleton]
    joint_names = d['names']
    edges = d['edges']
    return JointInfo(joint_names, edges)


def pick_indices(iterable, indices):
    it = iter(enumerate(iterable))
    i_now = None
    item = None

    for i_wanted in indices:
        while True:
            if i_now == i_wanted:
                yield item
                break
            i_now, item = next(it)


if __name__ == '__main__':
    main()
