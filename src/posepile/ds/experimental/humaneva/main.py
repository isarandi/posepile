import argparse
import os.path as osp
import re

import boxlib
import cameralib
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example

HUMANEVA_ROOT = f'{DATA_ROOT}/humaneva'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()  # TODO: complete this part


@spu.picklecache('humaneva_stage1.pkl', min_time="2021-12-29T05:09:32")
def make_stage1():
    names = 'pelv,thor,lsho,lelb,lwri,rsho,relb,rwri,lhip,lkne,lank,rhip,rkne,rank,head'
    edges = 'head-thor-pelv,lwri-lelb-lsho-thor-rsho-relb-rwri,lank-lkne-lhip-pelv-rhip-rkne-rank'
    joint_info = JointInfo(names, edges)
    j = joint_info.ids

    gt_data = spu.load_pickle(f'{HUMANEVA_ROOT}/gt_coords.pkl')
    detections_all = spu.load_pickle(f'{HUMANEVA_ROOT}/yolov4_detections.pkl')
    train_examples = []
    test_examples = []
    gt_per_video = spu.groupby(gt_data, lambda x: x['video'])

    with spu.ThrottledPool() as pool:
        for video_relpath, frames_data in gt_per_video.items():
            m = re.match(
                r'S(?P<subj>\d)/Image_Data/(?P<action>.+)_1_\(C(?P<cam>\d)\)\.avi', video_relpath)
            activity = m['action']
            subj = m['subj']
            cam = m['cam']
            camera = load_camera(f'{HUMANEVA_ROOT}/S{subj}/Calibration_Data/C{cam}.cal')

            frames_data = sorted(frames_data, key=lambda x: x['i_frame'])
            sampler = AdaptivePoseSampler(100)
            for frame_data in frames_data:
                world_coords = frame_data['world_coords']
                is_train = frame_data['is_train']
                i_frame = frame_data['i_frame']
                if ((not np.isfinite(world_coords).all()) or
                        (is_train and sampler.should_skip(world_coords))):
                    continue

                cam_coords = camera.world_to_camera(world_coords)
                is_corrupt = cam_coords[j.thor, 1] < cam_coords[j.head, 1]
                if is_corrupt:
                    world_coords[j.head] = np.nan

                image_path = (f'{HUMANEVA_ROOT}/S{subj}/Image_Data/{activity}_1_(C{cam})/'
                              f'frame_{i_frame:06d}.jpg')
                image_relpath = osp.relpath(image_path, DATA_ROOT)
                if not osp.exists(image_path):
                    print(f'Warn - Image not found {image_relpath}')
                    continue

                gt_box = get_gt_box(camera.world_to_image(world_coords))
                detections = detections_all[osp.relpath(image_path, HUMANEVA_ROOT)]
                if is_train and detections.size > 0:
                    i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in detections])
                    box = detections[i_det][:4]
                else:
                    box = gt_box

                ex = ds3d.Pose3DExample(
                    image_relpath, world_coords, box, camera, activity_name=activity)
                new_image_replath = (
                        'humaneva_downscaled/' +
                        osp.relpath(image_path, HUMANEVA_ROOT))
                container = train_examples if is_train else test_examples
                pool.apply_async(
                    make_efficient_example, (ex, new_image_replath), callback=container.append)

    train_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ds3d.Pose3DDataset(joint_info, train_examples, test_examples=test_examples)


def get_gt_box(imcoords):
    box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.1)
    if np.any(np.isnan(imcoords)):
        height = box[3]
        offset = height * 0.25
        box[1] -= offset
        box[3] += offset

    return box


def load_camera(path):
    values = [float(v) for v in spu.read_lines(path) if v]
    f, c, skew, distortion, R, t = np.split(values, (2, 4, 5, 10, 19))
    intrinsic_matrix = np.array([[f[0], skew[0], c[0]], [0, f[1], c[1]], [0, 0, 1]], np.float32)
    return cameralib.Camera(
        intrinsic_matrix=intrinsic_matrix, rot_world_to_cam=R.reshape(3, 3), trans_after_rot=t,
        distortion_coeffs=distortion)


if __name__ == '__main__':
    main()
