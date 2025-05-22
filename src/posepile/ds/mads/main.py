import argparse
import glob
import os.path as osp

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import tensorflow_hub as tfhub
import transforms3d.axangles
from posepile import util
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    parser.add_argument('--calibrate-cameras', action=spu.argparse.BoolAction)
    parser.add_argument('--generate-dataset', action=spu.argparse.BoolAction)
    parser.add_argument('--detector-path', type=str,
                        default='https://github.com/isarandi/tensorflow-yolov4-tflite/releases'
                                '/download/v0.1.0/yolov4_416.tar.gz')
    spu.initialize(parser)
    if FLAGS.stage == 1:
        make_stage1()
    else:
        make_dataset()


@spu.picklecache('mads_stage1.pkl', min_time="2021-12-17T01:53:15")
def make_stage1():
    root = f'{DATA_ROOT}/mads'
    joint_names = (
        'neck,pelv,lhip,lkne,lank,ltoe,rhip,rkne,rank,rtoe,lsho,'
        'lelb,lwri,lhan,rsho,relb,rwri,rhan,head'.split(','))
    names15 = (
        'neck,pelv,lhip,lkne,lank,rhip,rkne,rank,lsho,lelb,lwri,rsho,relb,rwri,head'.split(','))
    indices15 = [joint_names.index(name) for name in names15]

    edges = 'head-neck-pelv-rhip-rkne-rank-rtoe,neck-rsho-relb-rwri-rhan'
    joint_info = JointInfo(joint_names, edges)

    video_filepaths = sorted(glob.glob(f'{root}/multi_view_data/*/*.avi'))
    detector = tfhub.load(FLAGS.detector_path)
    examples = []

    with spu.ThrottledPool() as pool:
        for video_path in video_filepaths:
            direc = osp.dirname(video_path)
            i_cam = int(video_path[-5])
            camera = load_camera(f'{direc}/Calib_Cam{i_cam}.mat')
            video_world_coords = util.load_mat(video_path[:-6] + 'GT.mat').GTpose2
            pose_sampler = AdaptivePoseSampler(100, check_validity=True)

            with imageio.get_reader(video_path, 'ffmpeg') as frames:
                for i_frame, (frame, raw_coords) in enumerate(zip(frames, video_world_coords)):
                    if np.all(np.isnan(raw_coords)):
                        continue

                    if raw_coords.shape[0] == 15:
                        world_coords = np.full([19, 3], fill_value=np.nan, dtype=np.float32)
                        world_coords[indices15] = raw_coords
                    else:
                        world_coords = raw_coords

                    if pose_sampler.should_skip(world_coords):
                        continue

                    detections = detector.predict_single_image(frame, 0.2, 0.7).numpy()
                    imcoords = camera.world_to_image(world_coords)
                    gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.2)
                    if detections.size > 0:
                        i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in detections])
                        box = detections[i_det][:4]
                    else:
                        box = gt_box

                    new_image_replath = (
                            'mads_downscaled/' + osp.relpath(video_path, root) +
                            f'/{i_frame:06d}.jpg')
                    ex = ds3d.Pose3DExample(frame, world_coords, bbox=box, camera=camera)
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_replath), callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, examples)


# Stage3: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('mads.pkl', min_time="2021-12-04T20:56:48")
def make_dataset():
    return ds3d.add_masks(make_stage1(), f'{DATA_ROOT}/mads_downscaled/masks', 4)


def load_camera(path):
    data = util.load_mat(path)
    angle = np.linalg.norm(data.om_ext)
    R = transforms3d.axangles.axangle2mat(data.om_ext, angle)
    optical_center = -R.T @ data.T_ext
    return cameravision.Camera(
        rot_world_to_cam=R, optical_center=optical_center,
        intrinsic_matrix=data.KK, distortion_coeffs=data.kc, world_up=(0, 1, 0))


if __name__ == '__main__':
    main()
