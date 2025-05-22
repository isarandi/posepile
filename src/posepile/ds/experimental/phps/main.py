import argparse
import glob
import os
import os.path as osp
import re

import boxlib
import cameralib
import numpy as np
import simplepyutils as spu
import transforms3d
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.ds.experimental.ntu.recover_scale import find_up_vector
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example

PHPS_ROOT = f'{DATA_ROOT}/phps'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    parser.add_argument('--estimate-up-vectors', action=spu.argparse.BoolAction)
    spu.initialize(parser)

    if FLAGS.estimate_up_vectors:
        estimate_up_vectors()

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/phps_downscaled/tasks/task_result_{i_task:03d}.pkl'

    image_paths_all = sorted(glob.glob(f'{PHPS_ROOT}/color/*/*/*.jpg'))
    image_paths_by_seq_all = list(spu.groupby(image_paths_all, get_seq_id).items())
    image_paths_by_seq = image_paths_by_seq_all[i_task * 2:(i_task + 1) * 2]
    cameras = load_cameras_per_subject()

    detections_all = spu.load_pickle(f'{PHPS_ROOT}/yolov4_detections.pkl')
    examples = []
    seq_pattern = r'view(?P<cam>\d)/subject(?P<subj>\d{2})_group\d_time\d'

    with spu.ThrottledPool() as pool:
        for seq_id, image_paths in spu.progressbar(image_paths_by_seq):
            if 'demo' in seq_id:
                continue

            m = re.match(seq_pattern, seq_id)
            i_cam = int(m['cam']) - 1
            subj_id = int(m['subj'])
            camera = cameras[subj_id][i_cam]
            image_paths = sorted(image_paths, key=get_i_frame)
            world_coords_all = load_poses(f'{PHPS_ROOT}/pose/{seq_id.split("/")[1]}/pose.txt')
            pose_sampler = AdaptivePoseSampler2(100, True, True, 100)

            for image_path in image_paths:
                i_frame = get_i_frame(image_path)
                try:
                    world_coords = world_coords_all[i_frame]
                except IndexError:
                    continue

                if pose_sampler.should_skip(world_coords):
                    # print('skip', image_path)
                    continue

                imcoords = camera.world_to_image(world_coords)
                n_joints_in_frame = np.count_nonzero(
                    boxlib.contains(boxlib.full([1920, 1080]), imcoords))
                if n_joints_in_frame < 4:
                    continue

                gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.2)
                detections = detections_all[osp.relpath(image_path, f'{PHPS_ROOT}/color')]
                if detections.size > 0:
                    detections = [hflip_box(box) for box in detections]
                    i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in detections])
                    box = detections[i_det][:4]
                else:
                    box = gt_box

                if boxlib.iou(box, gt_box) < 0.3:
                    continue

                image_replath = osp.relpath(image_path, DATA_ROOT)
                new_image_replath = f'phps_downscaled/{osp.relpath(image_path, PHPS_ROOT)}'
                ex = ds3d.Pose3DExample(image_replath, world_coords, bbox=box, camera=camera)
                pool.apply_async(
                    make_efficient_example, (ex, new_image_replath), dict(horizontal_flip=True),
                    callback=examples.append)

    joint_info = get_joint_info()
    examples.sort(key=lambda ex: ex.image_path)
    ds = ds3d.Pose3DDataset(joint_info, examples)
    spu.dump_pickle(ds, out_path)


def hflip_box(box):
    W = 1920
    x, y, w, h, c = box
    return np.array([W - (x + w), y, w, h, c], np.float32)


def get_joint_info():
    return JointInfo(
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan',
        'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe')


def get_i_frame(path):
    return int(path.split('_')[-1].split('.')[0])


def get_seq_id(path):
    dirname = osp.dirname(path)
    return osp.relpath(dirname, f'{DATA_ROOT}/phps/color')


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('phps.pkl', min_time="2021-12-04T20:56:48")
def make_dataset():
    partial_paths = sorted(glob.glob(f'{DATA_ROOT}/phps_downscaled/tasks/task_result_*.pkl'))
    partial_dss = [spu.load_pickle(p) for p in partial_paths]
    main_ds = partial_dss[0]
    for ds in partial_dss[1:]:
        main_ds.examples[0].extend(ds.examples[0])

    ds3d.add_masks(main_ds, f'{DATA_ROOT}/phps_downscaled/masks', 4)
    return main_ds


def load_poses(path):
    arr = np.loadtxt(path)
    i_frames = arr[:, 0].astype(np.int32)
    n_frames = np.max(i_frames) + 1
    poses = np.full([n_frames, 24, 3], dtype=np.float32, fill_value=np.nan)
    poses[i_frames] = arr[:, 1:].reshape(-1, 24, 3) * 1000
    return poses


def get_extrinsic_matrix(params):
    quat, trans = np.split(params, [4])
    R = transforms3d.quaternions.quat2mat(quat)
    return np.block([[R, np.expand_dims(trans, -1)], [0, 0, 0, 1]])


def load_cameras(path):
    cam_data = spu.load_pickle(path)
    cameras = []
    for i_cam in range(3):
        # The distortion coefficients don't seem correct based on visual inspection of
        # undistorted images, so we ignore them.
        fx, fy, cx, cy, _, _, _ = cam_data[f'param_c{i_cam + 1}']
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        extr1 = get_extrinsic_matrix(cam_data[f'cd{i_cam + 1}'])
        extr2 = get_extrinsic_matrix(cam_data[f'd{i_cam + 1}p'])
        extrinsic_matrix = extr1 @ extr2
        cam = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)
        cameras.append(cam)
    return cameras


def load_cameras_per_subject():
    cameras1 = load_cameras(f'{PHPS_ROOT}/CamParams0906.pkl')
    cameras2 = load_cameras(f'{PHPS_ROOT}/CamParams0909.pkl')

    up1, up2 = np.load(f'{PHPS_ROOT}/world_up_vector.npy')
    for c in cameras1:
        c.world_up = up1

    for c in cameras2:
        c.world_up = up2

    result = {}
    for subj_id in [4, 5, 6, 9, 11, 12]:
        result[subj_id] = cameras1

    for subj_id in [1, 2, 3, 7, 8, 10]:
        result[subj_id] = cameras2

    return result


def estimate_up_vectors():
    pose_paths = glob.glob(f'{PHPS_ROOT}/pose/*/pose.txt')

    def get_i_calib(path):
        subj_id = int(osp.basename(osp.dirname(path)).split('_')[0][-2:])
        return 0 if subj_id in [4, 5, 6, 9, 11, 12] else 1

    pose_paths_per_calib = spu.groupby(pose_paths, get_i_calib)
    j = get_joint_info().ids
    up_vectors = {
        calib_id: find_up_vector(np.concatenate([
            load_poses(p)[:, [j.ltoe, j.rtoe]].reshape(-1, 3)
            for p in paths], axis=0))
        for calib_id, paths in pose_paths_per_calib.items()}

    up_vectors = [up_vectors[0], up_vectors[1]]
    print(up_vectors)
    np.save(f'{PHPS_ROOT}/world_up_vector.npy', up_vectors)


if __name__ == '__main__':
    main()
