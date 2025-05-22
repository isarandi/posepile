import argparse
import glob
import os
import os.path as osp
import random

import cameralib
import imageio.v2 as imageio
import numpy as np
import rlemasklib
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
import posepile.util.maskproc as maskproc
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.util import geom3d

NTU_ROOT = f'{DATA_ROOT}/ntu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--calibrate-intrinsics', action=spu.argparse.BoolAction)
    spu.initialize(parser)

    if FLAGS.calibrate_intrinsics:
        calibrate_intrinsics()

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_stage2()
    elif FLAGS.stage == 3:
        make_dataset()


@spu.picklecache('ntu_stage1.pkl', min_time="2021-12-19T16:10:03")
def make_stage1():
    skeleton_files = glob.glob(f'{NTU_ROOT}/nturgb+d_skeletons/*.skeleton')
    examples = []
    with spu.ThrottledPool() as pool:
        for p in spu.progressbar(skeleton_files):
            pool.apply_async(load_examples, (p,), callback=examples.extend)
    return examples


def make_stage2():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/ntu_downscaled/tasks/task_result_{i_task:03d}.pkl'
    # if osp.exists(out_path):
    #    return

    joint_info_base = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    i_selected_joints = [
        i for i, name in enumerate(joint_info_base.names)
        if name.endswith(('_cmu_panoptic', '_coco')) or '_' not in name]
    joint_info = joint_info_base.select_joints(i_selected_joints)
    joint_info.update_names([x.replace('cmu_panoptic', 'coco') for x in joint_info.names])

    ref_bone_len = np.array(
        spu.load_pickle(f'{NTU_ROOT}/predictor_bone_length_prior.pkl'), np.float32)
    pred_paths_all = spu.sorted_recursive_glob(f'{NTU_ROOT}/triang_scaled/**/*.pkl')
    # pred_paths_all = [p for p in pred_paths_all if 'S002C001P007R001A025' in p]
    print(len(pred_paths_all))
    pred_paths = pred_paths_all[i_task * 500:(i_task + 1) * 500]
    examples = []

    with spu.ThrottledPool() as pool:
        for pred_path in spu.progressbar(pred_paths):
            video_id = osp.basename(pred_path).split('_')[0]
            v = video_id
            video_masks = spu.load_pickle(f'{NTU_ROOT}/stcn_pred/{v[:4]}/{v[4:12]}/{v}_rgb.pkl')
            data = spu.load_pickle(pred_path)
            camera = data['camera']

            video_world_coords = mask_bad(data['poses3d'], data['errors'], 35)
            video_boxes = data['boxes']
            n_people = data['poses3d'].shape[1]
            pose_samplers = [AdaptivePoseSampler2(100, True, True, 100) for _ in range(n_people)]

            with imageio.get_reader(
                    f'{NTU_ROOT}/nturgb+d_rgb/{v[:4]}/{v[4:12]}/{v}_rgb.avi', 'ffmpeg') as frames:
                for i_frame, (
                        frame, world_coords_per_person, box_per_person, mask_per_person
                ) in enumerate(zip(frames, video_world_coords, video_boxes, video_masks)):
                    mask_union = rlemasklib.union(mask_per_person)
                    mask_union = maskproc.resize_mask(mask_union, frame.shape)
                    for i_person, (world_coords, box, pose_sampler) in enumerate(
                            zip(world_coords_per_person, box_per_person, pose_samplers)):
                        if not np.all(geom3d.are_bones_plausible(
                                world_coords, ref_bone_len, joint_info_base,
                                relsmall_thresh=0.3, relbig_thresh=1.5, absbig_thresh=150)):
                            continue

                        if pose_sampler.should_skip(world_coords):
                            continue

                        new_image_replath = (
                            f'ntu_downscaled/{v[:4]}/{v[4:12]}/'
                            f'{video_id}_{i_frame:06d}_{i_person}.jpg')
                        ex = ds3d.Pose3DExample(
                            frame, world_coords[i_selected_joints], bbox=box, camera=camera,
                            mask=mask_union)
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_replath),
                            callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    ds_partial = ds3d.Pose3DDataset(joint_info, examples)

    spu.dump_pickle(ds_partial, out_path)


@spu.picklecache('ntu.pkl', min_time="2022-02-08T19:46:54")
def make_dataset():
    partial_paths = spu.sorted_recursive_glob(
        f'{DATA_ROOT}/ntu_downscaled/tasks/task_result_*.pkl')
    partial_dss = [spu.load_pickle(p) for p in partial_paths]
    main_ds = partial_dss[0]
    for ds in partial_dss[1:]:
        main_ds.examples[0].extend(ds.examples[0])
    ds3d.filter_dataset_by_plausibility(
        main_ds, relsmall_thresh=0.5, relbig_thresh=1.25, absbig_thresh=80)
    ds3d.add_masks(main_ds, f'{DATA_ROOT}/ntu_downscaled/masks', 3)
    return main_ds


def mask_bad(triang, err, thresh=35):
    triang = triang.copy()
    triang[err > thresh] = np.nan
    return triang


def calibrate_intrinsics():
    examples = make_stage1()
    grouped = spu.groupby(examples, lambda ex: ex['video_id'][:8])
    cameras = {k: calibrate_intrinsics_for_examples(exs) for k, exs in grouped.items()}
    video_paths = spu.sorted_recursive_glob(f'{NTU_ROOT}/nturgb+d_rgb/**/*.avi')

    video_to_camera = {}
    for video_path in video_paths:
        p = osp.relpath(video_path, f'{NTU_ROOT}/nturgb+d_rgb')
        video_to_camera[p] = cameras[p[:8]].intrinsic_matrix
    spu.dump_pickle(video_to_camera, f'{NTU_ROOT}/video_to_camera.pkl')
    spu.dump_pickle(cameras, f'{NTU_ROOT}/cameras.pkl')


def load_examples(path):
    nums = [line.split(' ') for line in spu.read_lines(path)]
    i = 0
    video_id = osp.basename(path).split('.')[0]

    def read():
        nonlocal i
        result = nums[i]
        i += 1
        return result

    def read_one_int():
        return int(read()[0])

    examples = []
    n_frames = read_one_int()
    for i_frame in range(n_frames):
        n_bodies = read_one_int()
        for i_body in range(n_bodies):
            body_info = read()
            body_id = int(body_info[0])
            n_joints = read_one_int()
            camcoords = np.empty([n_joints, 3], dtype=np.float32)
            coords2d = np.empty([n_joints, 2], dtype=np.float32)
            track_state = np.empty([n_joints], dtype=np.int32)
            for i_joint in range(n_joints):
                line = [float(x) for x in read()]
                x3, y3, z3, xd, yd, x2, y2, qw, qx, qy, qz, t = line
                camcoords[i_joint] = [x3 * 1000, -y3 * 1000, z3 * 1000]
                coords2d[i_joint] = [x2, y2]
                track_state[i_joint] = t
            examples.append(
                dict(video_id=video_id, i_frame=i_frame, body_id=body_id, coords3d=camcoords,
                     coords2d=coords2d, track_state=track_state))
    return examples


def calibrate_intrinsics_for_examples(examples):
    def single_try():
        used_examples = random.sample(examples, min(150, len(examples)))

        n_rows = 25 * len(used_examples) * 2
        A = np.empty((n_rows, 4), dtype=np.float32)
        b = np.empty((n_rows, 1), dtype=np.float32)
        i = 0
        for ex in used_examples:
            for coords2d, coords3d in zip(ex['coords2d'], ex['coords3d']):
                x, y = coords2d
                x3, y3, z3 = coords3d
                A[i] = [x3 / z3, 0, 1, 0]
                A[i + 1] = [0, y3 / z3, 0, 1]
                b[i] = [x]
                b[i + 1] = [y]
                i += 2

        rms_A = np.sqrt(np.mean(np.square(A), axis=0))
        rms_b = np.sqrt(np.mean(np.square(b), axis=0))
        result, residual, *_ = np.linalg.lstsq(A / rms_A, b / rms_b, rcond=None)
        result = result[:, 0] * rms_b / rms_A
        fx, fy, cx, cy = result
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return cameralib.Camera(intrinsic_matrix=intrinsics, world_up=(0, -1, 0))

    # We need to try the calibration multiple times
    failed_at_least_once = False
    for _ in range(10):
        try:
            result = single_try()
            if failed_at_least_once:
                print('Finally it worked!')
            return result
        except (np.linalg.LinAlgError, ValueError):
            print('Error, but retrying...')
            failed_at_least_once = True
            pass
    else:
        print('Failed!')
        raise np.linalg.LinAlgError()


def get_video_id(path):
    return osp.basename(path).split('_')[0]


def get_seq_id(video_id):
    return video_id[:4] + video_id[8:]


def get_cam_id(video_id):
    return video_id[4:8]


def examples_to_tracks(exs):
    n_frames = max(x['i_frame'] for x in exs) + 1
    gt3d = [[] for _ in range(n_frames)]
    for ex in exs:
        i_frame = int(ex['i_frame'])
        gt3d[i_frame].append(ex['coords3d'])
    gt3d = [np.array(a, np.float32).reshape([-1, 25, 3]) for a in gt3d]
    return gt3d


def get_kinect_joint_info():
    names = (
        'pelv,bell,head,htop,rsho,relb,rwri,rthu,lsho,lelb,lwri,lthu,rhip,rkne,rank,rtoe,lhip,'
        'lkne,lank,ltoe,neck,rhan1,rhan2,lhan1,lhan2')
    edges = 'htop-head-neck-bell-pelv-rhip-rkne-rank-rtoe,neck-rsho-relb-rwri-rthu-rhan1-rhan2'
    return JointInfo(names, edges)


if __name__ == '__main__':
    main()
