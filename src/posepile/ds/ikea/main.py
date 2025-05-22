import argparse
import os.path as osp

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.util.preproc_for_efficiency import make_efficient_example

IKEA_ROOT = f'{DATA_ROOT}/ikea'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--extract-annotated-frames', action=spu.argparse.BoolAction)
    spu.initialize(parser)

    if FLAGS.extract_annotated_frames:
        extract_annotated_frames()

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def extract_annotated_frames():
    pose_paths_all = get_pose_paths()
    videos_all = list(spu.groupby(pose_paths_all, lambda p: osp.dirname(p)).items())
    # i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    for dirname, pose_paths in spu.progressbar(videos_all):
        print(dirname)
        parts = spu.split_path(dirname)
        seq_id = parts[-4] + '/' + parts[-3] + '/' + parts[-2]
        video_path = f'{IKEA_ROOT}/{seq_id}/images/scan_video.avi'
        i_annotated_frames = [int(osp.splitext(osp.basename(p))[0]) for p in pose_paths]

        with imageio.get_reader(video_path, 'ffmpeg') as reader:
            for i_frame, frame in enumerate(reader):
                if i_frame in i_annotated_frames:
                    dst_path = f'{osp.dirname(video_path)}/{i_frame:06d}.jpg'
                    imageio.imwrite(dst_path, frame, quality=95)


@spu.picklecache('ikea_stage1.pkl', min_time="2022-01-19T01:56:31")
def make_stage1():
    video_filepaths = get_video_paths()
    cameras = load_cameras()
    detections_all = spu.load_pickle(f'{IKEA_ROOT}/yolov4_detections.pkl')
    examples = []

    with spu.ThrottledPool() as pool:
        for video_path in spu.progressbar(video_filepaths):
            parts = spu.split_path(video_path)
            cam_id = parts[-3]
            seq_id = parts[-5] + '/' + parts[-4]

            if parts[-4] == 'special_test':
                continue
            calib_id = parts[-4].split('_')[3]
            camera = cameras[calib_id][cam_id]

            pose_paths = spu.sorted_recursive_glob(f'{IKEA_ROOT}/{seq_id}/{cam_id}/pose3d/*.json')
            frame_paths = spu.sorted_recursive_glob(f'{IKEA_ROOT}/{seq_id}/{cam_id}/images/*.jpg')
            pose_sampler = AdaptivePoseSampler(100, check_validity=True)

            for i_frame, (frame_path, pose_path) in enumerate(zip(frame_paths, pose_paths)):
                camcoords = load_pose(pose_path)
                world_coords = camera.camera_to_world(camcoords)
                if pose_sampler.should_skip(camcoords):
                    continue

                n_joints_in_frame = np.count_nonzero(
                    camera.is_visible(world_coords, imsize=[1920, 1080]))
                if n_joints_in_frame < 4:
                    continue

                imcoords = camera.world_to_image(world_coords)
                gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.2)
                detections = detections_all[osp.relpath(frame_path, IKEA_ROOT)]
                if detections.size > 0:
                    i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in detections])
                    box = detections[i_det][:4]
                else:
                    box = gt_box

                if boxlib.iou(box, gt_box) < 0.3:
                    continue

                frame_relpath = osp.relpath(frame_path, DATA_ROOT)
                new_image_replath = f'ikea_downscaled/{osp.relpath(frame_path, IKEA_ROOT)}'
                ex = ds3d.Pose3DExample(frame_relpath, world_coords, bbox=box, camera=camera)
                pool.apply_async(
                    make_efficient_example, (ex, new_image_replath), callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rank-rkne-rhip-lhip,rhip-rsho-relb-rwri,rsho-lsho,rear-reye-nose')
    return ds3d.Pose3DDataset(joint_info, examples)


@spu.picklecache('ikea_pose_paths.pkl', min_time="2021-12-17T01:53:15")
def get_pose_paths():
    return spu.sorted_recursive_glob(f'{IKEA_ROOT}/**/pose3d/*.json')


@spu.picklecache('ikea_video_paths.pkl', min_time="2021-12-17T01:53:15")
def get_video_paths():
    return spu.sorted_recursive_glob(f'{IKEA_ROOT}/**/images/scan_video.avi')


def load_pose(pose_path):
    pose = spu.load_json(pose_path)
    pose = np.array(pose['pose_keypoints_3d'], np.float32).reshape(-1, 4)[:, :3] * 10
    pose[pose == 0] = np.nan
    return pose


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('ikea.pkl', min_time="2022-01-19T02:01:04")
def make_dataset():
    return ds3d.add_masks(make_stage1(), f'{DATA_ROOT}/ikea_downscaled/masks', 5)


def load_cameras():
    data = spu.load_pickle(f'{IKEA_ROOT}/Calibration/camera_parameters.pkl')
    # The world_up vector is not given in the dataset, we set it to one camera's vertical axis.
    return {
        k1: {
            k2: cameravision.Camera(
                intrinsic_matrix=v2['K'], rot_world_to_cam=v2['R'], world_up=(0, -1, 0))
            for k2, v2 in v1.items()}
        for k1, v1 in data.items()}


if __name__ == '__main__':
    main()
