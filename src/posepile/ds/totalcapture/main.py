import argparse
import glob
import os
import os.path as osp
import re

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS

TOTALCAPTURE_ROOT = f'{DATA_ROOT}/totalcapture'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    cameras = load_cameras(f'{TOTALCAPTURE_ROOT}/calibration.cal')
    video_paths = spu.sorted_recursive_glob(f'{TOTALCAPTURE_ROOT}/video/**/*.mp4')
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    video_path = video_paths[i_task]
    print(video_path)
    m = re.search(
        r'TC_S(?P<subj>\d)_(?P<action>.+)(?P<rep>\d)_cam(?P<cam>\d).mp4',
        osp.basename(video_path))
    subj = m['subj']
    action = m['action']
    rep = m['rep']
    is_test = f'{action}{rep}' in 'walking2 freestyle3 acting3'.split() or subj in ('s4', 's5')
    # todo: Check for exact test protocol in literature (e.g., frame selection)
    if is_test:
        return

    i_cam = int(m['cam']) - 1
    mask_path = video_path.replace('/video/', '/mattes/')
    camera = cameras[i_cam]
    poses = load_joints(f'{TOTALCAPTURE_ROOT}/vicon/S{subj}/{action}{rep}/gt_skel_gbl_pos.txt')

    examples = []
    with imageio.get_reader(video_path, 'ffmpeg') as video, \
            imageio.get_reader(mask_path, 'ffmpeg') as mask_video:
        sampler = AdaptivePoseSampler(100)
        for i_frame, (frame, mask, world_coords) in enumerate(zip(video, mask_video, poses)):
            enough_visible = np.count_nonzero(
                camera.is_visible(world_coords, [frame.shape[1], frame.shape[0]])) >= 6
            if not enough_visible or sampler.should_skip(world_coords):
                continue

            im_coords = camera.world_to_image(world_coords)
            bbox_joints = boxlib.expand(boxlib.bb_of_points(im_coords), 1.25)
            bbox_mask = boxlib.bb_of_mask(mask[..., 0])
            bbox = boxlib.intersection(
                boxlib.full(imshape=frame.shape), boxlib.intersection(bbox_joints, bbox_mask))
            ex = ds3d.Pose3DExample(frame, world_coords, bbox, camera, mask=mask[..., 0])
            video_relpath_noext = osp.splitext(
                osp.relpath(video_path, TOTALCAPTURE_ROOT))[0]
            new_image_relpath = f'totalcapture_downscaled/{video_relpath_noext}/{i_frame:06d}.jpg'
            examples.append(make_efficient_example(ex, new_image_relpath))
    spu.dump_pickle(
        examples, f'{DATA_ROOT}/totalcapture_downscaled/stage1/examples_{i_task:06d}.pkl')


@spu.picklecache('totalcapture.pkl', min_time="2021-12-05T21:47:55")
def make_dataset():
    names = (
        'pelv,spin,spin1,spin2,spin3,neck,head,rcla,rsho,relb,rwri,lcla,lsho,lelb,lwri,rhip,rkne,'
        'rank,lhip,lkne,lank')
    edges = 'pelv-spin-spin1-spin2-spin3-neck-head,neck-rcla-rsho-relb-rwri-rhan,' \
            'pelv-rhip-rkne-rank'
    joint_info = JointInfo(names, edges)
    example_paths = glob.glob(f'{DATA_ROOT}/totalcapture_downscaled/stage1/examples_*.pkl')
    examples = [ex for p in example_paths for ex in spu.load_pickle(p)
                if '/s4/' not in ex.image_path and '/s5/' not in ex.image_path]
    return ds3d.Pose3DDataset(joint_info, examples)


def load_joints(path):
    lines = spu.read_lines(path)[1:]
    coords_inch = np.array([
        [[float(coord)
          for coord in point.split(' ')]
         for point in line.split('\t') if point]
        for line in lines], np.float32)
    return coords_inch * 25.4  # convert inch to mm


def load_cameras(path):
    cameras = []
    lines_all = spu.read_lines(path)[1:]
    for i_start in range(0, len(lines_all), 7):
        fx, fy, cx, cy = [float(x) for x in lines_all[i_start + 1].split()]
        dist_coeffs = np.zeros(5, np.float32)
        dist_coeffs[0] = float(lines_all[i_start + 2])
        R = np.array([
            [float(x)
             for x in line.split()]
            for line in lines_all[i_start + 3:i_start + 6]], np.float32)
        t = np.array([float(x) for x in lines_all[i_start + 6].split()], np.float32)
        intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        cam = cameralib.Camera(
            intrinsic_matrix=intr, rot_world_to_cam=R, trans_after_rot=1000 * t,
            distortion_coeffs=dist_coeffs, world_up=(0, 1, 0))
        cameras.append(cam)
    return cameras


if __name__ == '__main__':
    main()
