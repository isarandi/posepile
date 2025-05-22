import argparse
import glob
import itertools
import os
import os.path as osp
import re

import boxlib
import cameravision
import cv2
import ffmpeg
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_stage2()
    elif FLAGS.stage == 3:
        make_dataset()


def make_stage1():
    joint_names = ('nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,'
                   'rwri,lhip,rhip,lkne,rkne,lank,rank,neck,pelv')
    edges = 'nose-neck-pelv-rhip-rkne-rank,rwri-relb-rsho-neck,leye-reye-rear'
    joint_info = JointInfo(joint_names, edges)
    j = joint_info.ids
    aist_root = f'{DATA_ROOT}/aist'

    camera_mapping = dict(
        [l.split() for l in spu.read_lines(f'{aist_root}/annotations/cameras/mapping.txt')])
    camera_settings = {
        spu.path_stem(p): load_aist_cameras(p)
        for p in glob.glob(f'{aist_root}/annotations/cameras/*.json')}
    ignore_list = spu.read_lines(f'{aist_root}/annotations/ignore_list.txt')

    keypoint_paths = glob.glob(f'{aist_root}/annotations/keypoints3d/*.pkl')
    examples = []
    for keypoint_path in spu.progressbar(keypoint_paths):
        multicam_id = spu.path_stem(keypoint_path)
        if multicam_id in ignore_list:
            continue
        multicam_id_wildcard = multicam_id.replace('cAll', 'c*')
        video_paths = glob.glob(f'{aist_root}/videos/{multicam_id_wildcard}.mp4')
        cameras = camera_settings[camera_mapping[multicam_id]]
        seq_coords = spu.load_pickle(keypoint_path)['keypoints3d_optim'] * 10
        neck = np.mean(seq_coords[:, [j.lsho, j.rsho]], axis=1, keepdims=True)
        pelv = np.mean(seq_coords[:, [j.lhip, j.rhip]], axis=1, keepdims=True)
        seq_coords = np.concatenate([seq_coords, neck, pelv], axis=1)

        for video_path in video_paths:
            video_relpath = osp.relpath(video_path, DATA_ROOT)
            i_cam = int(re.search(r'_c(\d{2})_', video_path).group(1))
            camera = cameras[i_cam - 1]
            pose_sampler = AdaptivePoseSampler(100)
            for i_frame, world_pose in enumerate(seq_coords):
                if pose_sampler.should_skip(world_pose):
                    continue
                box = boxlib.expand(boxlib.bb_of_points(camera.world_to_image(world_pose)), 1.15)
                image_relpath = f'{video_relpath}/{i_frame:06d}'
                ex = ds3d.Pose3DExample(image_relpath, world_pose, box, camera)
                examples.append(ex)

    def get_video_path(ex):
        return osp.dirname(ex.image_path)

    examples_by_video = spu.groupby(examples, get_video_path)
    for i, (video_path, vid_examples) in enumerate(examples_by_video.items()):
        spu.dump_pickle(
            (video_path, vid_examples), f'{DATA_ROOT}/aist_downscaled/tasks/task_{i:06d}.pkl')


def make_stage2():
    downscaled_root = f'{DATA_ROOT}/aist_downscaled'
    task_paths = spu.sorted_recursive_glob(f'{downscaled_root}/tasks/task_*.pkl')
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    task_path = task_paths[task_id]
    video_path, examples = spu.load_pickle(task_path)
    output_examples = []

    def get_i_frame(ex):
        return int(osp.basename(ex.image_path))

    i_frame_to_examples = spu.groupby(examples, get_i_frame)
    frames = ffmpeg_video_read(f'{DATA_ROOT}/{video_path}')
    for i_frame, frame in enumerate(frames):
        if i_frame not in i_frame_to_examples:
            continue
        ex = i_frame_to_examples[i_frame][0]
        ex.image_path = frame
        video_id = spu.path_stem(video_path)
        new_image_relpath = f'aist_downscaled/videos/{video_id}/{i_frame:06d}.jpg'
        output_examples.append(make_efficient_example(ex, new_image_relpath, 0.8))

    spu.dump_pickle(
        (video_path, output_examples),
        f'{downscaled_root}/tasks/processed_task_{task_id:06d}.pkl')


@spu.picklecache('aist.pkl', min_time="2021-07-09T22:07:21")
def make_dataset():
    joint_names = 'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,' \
                  'rwri,lhip,rhip,lkne,rkne,lank,rank,neck,pelv'.split(',')
    edges = 'nose-neck-pelv-lhip-lkne-lank,rwri-relb-rsho-neck-lsho-lelb-lwri,' \
            'lear-leye-reye-rear,pelv-rhip-rkne-rank'
    joint_info = JointInfo(joint_names, edges)
    aist_root = f'{DATA_ROOT}/aist'
    downscaled_root = f'{DATA_ROOT}/aist_downscaled'
    train_seqs = spu.read_lines(f'{aist_root}/annotations/splits/pose_train.txt')
    val_seqs = spu.read_lines(f'{aist_root}/annotations/splits/pose_val.txt')
    test_seqs = spu.read_lines(f'{aist_root}/annotations/splits/pose_test.txt')
    train_examples = []
    val_examples = []
    test_examples = []
    task_paths = spu.sorted_recursive_glob(f'{downscaled_root}/tasks/processed_task*.pkl')

    for task_path in spu.progressbar(task_paths):
        video_path, examples = spu.load_pickle(task_path)
        video_id = spu.path_stem(video_path)
        i_cam = int(re.search(r'_c(\d{2})_', video_path).group(1))
        multicam_id = video_id.replace(f'c{i_cam:02d}', 'cAll')
        if multicam_id in train_seqs:
            train_examples.extend(examples)
        elif multicam_id in val_seqs:
            val_examples.extend(examples)
        elif multicam_id in test_seqs:
            test_examples.extend(examples)
        else:
            raise ValueError()

    mask_paths = glob.glob(f'{downscaled_root}/masks/*.pkl')
    mask_dict = {}
    for path in spu.progressbar(mask_paths):
        mask_dict.update(spu.load_pickle(path))

    for ex in itertools.chain(train_examples, val_examples, test_examples):
        relpath = osp.relpath(f'{DATA_ROOT}/{ex.image_path}', downscaled_root)
        if relpath in mask_dict:
            ex.mask = mask_dict[relpath]
        else:
            raise RuntimeError(f'{relpath} does not have a mask')

    return ds3d.Pose3DDataset(joint_info, train_examples, val_examples, test_examples)


def load_aist_cameras(path):
    data = spu.load_json(path)
    cameras = []
    for cam_data in data:
        rvec = np.array([cam_data['rotation']], np.float32)
        rot_matrix = cv2.Rodrigues(rvec)[0]
        tvec = np.array(cam_data['translation']) * 10
        camera = cameravision.Camera(
            optical_center=-rot_matrix.T @ tvec,
            rot_world_to_cam=rot_matrix,
            intrinsic_matrix=cam_data['matrix'],
            distortion_coeffs=cam_data['distortions'],
            world_up=(0, 1, 0))
        cameras.append(camera)
    return cameras


def ffmpeg_video_read(video_path, fps=60):
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    pipeline = ffmpeg.input(video_path)
    if fps is not None:
        pipeline = pipeline.filter('fps', fps=fps, round='up')
    pipeline = pipeline.output('pipe:', format='rawvideo', pix_fmt='rgb24')
    process = pipeline.run_async(pipe_stdout=True)

    while in_bytes := process.stdout.read(width * height * 3):
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        yield frame.copy()

    process.wait()


if __name__ == '__main__':
    main()
