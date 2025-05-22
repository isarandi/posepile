import os
import os.path as osp

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from scipy.spatial.transform import Rotation as R
import argparse
from simplepyutils import FLAGS
from metrabs_tf.improc import draw_stick_figure
from posepile.joint_info import JointInfo
DATASET_NAME = 'freeman'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    train_seqs = spu.read_lines(f'{DATASET_DIR}/train.txt')
    val_seqs = spu.read_lines(f'{DATASET_DIR}/valid.txt')
    camera_names = [f'c{i + 1:02d}' for i in range(8)]
    seqs = [*train_seqs, *val_seqs]
    seq_name = seqs[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    out_path = f'{DATA_ROOT}/freeman_downscaled/stage1/{seq_name}.pkl'
    if osp.exists(out_path):
        return
    examples = []
    with spu.ThrottledPool() as pool:
        cameras = load_cameras(f'{DATASET_DIR}/cameras/{seq_name}.json')

        kp3d = np.load(
            f'{DATASET_DIR}/keypoints3d/{seq_name}.npy',
            allow_pickle=True).item()['keypoints3d'] * 10

        for camera_name in camera_names:
            camera = cameras[camera_name]
            video_reader = imageio.get_reader(
                f'{DATASET_DIR}/videos/{seq_name}/vframes/{camera_name}.mp4')
            detections_seq = spu.load_pickle(
                f'{DATASET_DIR}/detections/videos/{seq_name}/vframes/{camera_name}.pkl')
            sampler = AdaptivePoseSampler2(
                100, True, True, 100)

            for i_frame, (im, detections) in enumerate(
                    zip(video_reader, spu.progressbar(
                        detections_seq, leave=False, desc=f'{seq_name}/{camera_name}'))):
                if len(detections) == 0:
                    continue

                joints = kp3d[i_frame]
                if sampler.should_skip(joints):
                    continue

                imcoords = camera.world_to_image(joints)
                bbox_gt = boxlib.expand(boxlib.bb_of_points(imcoords), 1.1)
                ious = [boxlib.iou(det, bbox_gt) for det in detections]
                if np.max(ious) < 0.5:
                    continue

                bbox_det = detections[np.argmax(ious)][:4]
                ex = ds3d.Pose3DExample(
                    image_path=im, camera=camera, world_coords=joints, bbox=bbox_det)
                new_image_relpath = (
                    f'freeman_downscaled/{seq_name}/{camera_name}/{i_frame:06d}.jpg')
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath),
                    kwargs=dict(extreme_perspective=True),
                    callback=examples.append)

    spu.dump_pickle(examples, out_path)


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    train_seqs = spu.read_lines(f'{DATASET_DIR}/train.txt')
    val_seqs = spu.read_lines(f'{DATASET_DIR}/valid.txt')

    examples_train = []
    examples_val = []

    for seq_name in spu.progressbar(train_seqs, desc='train'):
        examples_train.extend(spu.load_pickle(
            f'{DATA_ROOT}/freeman_downscaled/stage1/{seq_name}.pkl'))
    for seq_name in spu.progressbar(val_seqs, desc='val'):
        examples_val.extend(spu.load_pickle(
            f'{DATA_ROOT}/freeman_downscaled/stage1/{seq_name}.pkl'))

    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear')
    return ds3d.Pose3DDataset(joint_info, examples_train, examples_val)


def load_cameras(path):
    return {
        caminfo['name']: cameralib.Camera(
            intrinsic_matrix=caminfo['matrix'],
            rot_world_to_cam=R.from_rotvec(caminfo['rotation']).as_matrix(),
            trans_after_rot=np.array(caminfo['translation'], np.float32) * 10,
            #distortion_coeffs=caminfo['distortions'],
            world_up=(0, 0, -1)) for caminfo in spu.load_json(path)}


if __name__ == '__main__':
    main()
