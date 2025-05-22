import argparse
import glob
import itertools
import os
import os.path as osp

import boxlib
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
import posepile.ds.tdhp.main as tdhp_main
from posepile import util
from posepile.ds.muco.main import make_joint_info
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.util.preproc_for_efficiency import make_efficient_example

ROOT_3DHP = f'{DATA_ROOT}/3dhp_full'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/3dhp_full_downscaled/examples/examples_{i_task:06d}.pkl'
    if osp.exists(out_path):
        return

    tasks = list(itertools.product(range(8), range(2), range(14)))
    i_subj, i_seq, i_cam = tasks[i_task]

    joint_info, selected_joints = make_joint_info()
    detector = load_detector()
    num_frames = np.asarray(
        [[6416, 12430], [6502, 6081], [12488, 12283], [6171, 6675], [12820, 12312], [6188, 6145],
         [6239, 6320], [6468, 6054]])

    examples = []
    seqpath = f'{ROOT_3DHP}/S{i_subj + 1}/Seq{i_seq + 1}'
    cam3d_coords = [ann.reshape([ann.shape[0], -1, 3])[:, selected_joints]
                    for ann in util.load_mat(f'{seqpath}/annot.mat').annot3]
    camera = tdhp_main.load_cameras(f'{seqpath}/camera.calibration')[i_cam]
    n_frames = num_frames[i_subj, i_seq]

    if i_subj == 5 and i_seq == 1 and i_cam == 2:
        # This video is shorter for some reason
        n_frames = 3911

    video_path = f'{ROOT_3DHP}/S{i_subj + 1}/Seq{i_seq + 1}/imageSequence/video_{i_cam}.avi'
    mask_video_path = f'{ROOT_3DHP}/S{i_subj + 1}/Seq{i_seq + 1}/FGmasks/video_{i_cam}.avi'

    pose_sampler = AdaptivePoseSampler(100)
    with imageio.get_reader(video_path, 'ffmpeg') as video, \
            imageio.get_reader(mask_video_path, 'ffmpeg') as mask_video:
        for i_frame, frame, mask_frame in zip(spu.progressbar(range(n_frames)), video, mask_video):
            cam_coords = cam3d_coords[i_cam][i_frame]
            if pose_sampler.should_skip(cam_coords):
                continue

            world_coords = camera.camera_to_world(cam_coords)
            n_visible_joints = np.count_nonzero(camera.is_visible(world_coords, [2048, 2048]))
            if n_visible_joints < joint_info.n_joints // 3:
                continue

            im_coords = camera.camera_to_image(cam_coords)
            detections = detector.predict_single_image(frame, 0.2, 0.7).numpy()
            bbox = get_bbox(im_coords, detections)
            mask = (mask_frame[..., 0] > 32).astype(np.uint8) * 255
            new_image_relpath = (f'3dhp_full_downscaled/S{i_subj + 1}/Seq{i_seq + 1}/'
                                 f'imageSequence/video_{i_cam}/{i_frame:06d}.jpg')
            ex = ds3d.Pose3DExample(frame, world_coords, bbox, camera, mask=mask)
            examples.append(make_efficient_example(
                ex, new_image_relpath, image_adjustments_3dhp=True))

    spu.dump_pickle(examples, out_path)


@spu.picklecache('3dhp_full.pkl', min_time="2020-11-02T22:14:33")
def make_dataset():
    joint_info, _ = make_joint_info()
    example_paths = glob.glob(f'{DATA_ROOT}/3dhp_full_downscaled/examples/examples_*.pkl')
    examples = [ex for p in example_paths for ex in spu.load_pickle(p)]
    examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, examples)


def get_bbox(im_coords, detections):
    joint_box = boxlib.expand(boxlib.bb_of_points(im_coords), 1.05)
    if detections.size > 0:
        detection_box = max(detections, key=lambda x: x[4])[:4]
        union_box = boxlib.box_hull(detection_box, joint_box)
        # Sanity check
        if boxlib.iou(union_box, joint_box) > 0.5:
            return union_box
    return joint_box


def load_detector():
    import tensorflow_hub as tfhub
    return tfhub.load('https://github.com/isarandi/tensorflow-yolov4-tflite/releases'
                      '/download/v0.1.0/yolov4_416.tar.gz')


if __name__ == '__main__':
    main()
