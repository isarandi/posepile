import zipfile

import boxlib
import cameravision
import imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import os.path as osp
import smpl.numpy
import os
import re
import trimesh
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

DATASET_NAME = 'mri'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    make_dataset()


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    video_paths = spu.sorted_recursive_glob(f'{DATASET_DIR}/**/*0.mp4')
    examples = []
    with spu.ThrottledPool() as pool:
        for video_path in spu.progressbar(video_paths):
            video_relpath = osp.relpath(video_path, DATASET_DIR)
            match = re.match(
                r'subject(?P<subj>\d+)_color(?P<cam>\d)\.mp4', osp.basename(video_path))
            i_subj = match['subj']
            i_cam = int(match['cam'])

            data = spu.load_pickle(
                f'{DATASET_DIR}/dataset_release/aligned_data/pose_labels/'
                f'subject{i_subj}_all_labels.cpl')
            detections_seq = spu.load_pickle(
                f'{DATASET_DIR}/detections/' +
                video_relpath.replace('.mp4', '.pkl'))

            cam = cameravision.Camera(
                intrinsic_matrix=data['camera_matrix'][f'cam{i_cam + 1}_intrinsic'],
                extrinsic_matrix=data['camera_matrix'][f'cam{i_cam + 1}_extrinsic'],
                world_up=(0, 1, 0))
            cam.R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ cam.R
            cam.t *= 1000

            joints3d_seq = np.transpose(data['refined_gt_kps'], (0, 2, 1)) * 1000
            sampler = AdaptivePoseSampler2(100, True, True, 500)

            reader = imageio.get_reader(video_path)
            for i_frame, (image, world_coords, detections) in enumerate(
                    zip(reader, joints3d_seq, spu.progressbar(detections_seq, desc=video_relpath))):
                if (np.all(world_coords == 0) or
                        sampler.should_skip(world_coords) or
                        len(detections) == 0):
                    continue

                # the image is wrongly flipped horizontally in the original data
                detections[:, 0] = 1920 - (detections[:, 0] + detections[:, 2])
                detections = detections[:, :4]
                bbox_gt = boxlib.expand(boxlib.bb_of_points(cam.world_to_image(world_coords)), 1.1)
                ious = [boxlib.iou(bbox_gt, det) for det in detections]
                if np.max(ious) < 0.1:
                    continue
                bbox_det = detections[np.argmax(ious)]

                # the image is wrongly flipped horizontally in the original data
                ex = ds3d.Pose3DExample(
                    image_path=image[:, ::-1], camera=cam, world_coords=world_coords, bbox=bbox_det)

                outdir = osp.splitext(osp.basename(video_path))[0]
                new_image_relpath = f'mri_downscaled/{outdir}/{i_frame:06d}.jpg'
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath),
                    callback=examples.append)

    joint_info = ds3d.JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear')
    return ds3d.Pose3DDataset(joint_info, examples)


if __name__ == '__main__':
    main()
