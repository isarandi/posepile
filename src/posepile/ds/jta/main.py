import argparse
import collections
import glob
import os
import os.path as osp

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import scipy.optimize
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.util import geom3d
from simplepyutils import FLAGS

JTA_ROOT = f'{DATA_ROOT}/jta'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/jta_downscaled/tasks/task_result_{i_task:03d}.pkl'
    video_filepaths_all = spu.sorted_recursive_glob(f'{JTA_ROOT}/videos/*/*.mp4')
    video_path = video_filepaths_all[i_task]

    examples = []
    intr = np.array([[1167, 0, 960], [0, 1167, 540], [0, 0, 1]], np.float32)
    camera = cameravision.Camera(intrinsic_matrix=intr)

    joint_info = JointInfo(
        'htop,head,neck,rcla,rsho,relb,rwri,lcla,lsho,lelb,lwri,spin0,'
        'spin1,spin2,spin3,pelv,rhip,rkne,rank,lhip,lkne,lank',
        'rank-rkne-rhip-pelv-spin3-spin2-spin1-spin0-neck-head-htop,neck-rcla-rsho-relb-rwri')

    with spu.ThrottledPool() as pool:
        relpath = osp.relpath(video_path, f'{JTA_ROOT}/videos')
        poses_allframes = load_poses(
            spu.replace_extension(f'{JTA_ROOT}/annotations/{relpath}', '.json'))
        detections_allframes = spu.load_pickle(
            spu.replace_extension(f'{JTA_ROOT}/detections/{relpath}', '.pkl'))
        pose_samplers = collections.defaultdict(lambda: AdaptivePoseSampler(100))

        with imageio.get_reader(video_path, 'ffmpeg') as reader:
            for i_frame, (frame, poses, detections) in enumerate(
                    zip(reader, poses_allframes, detections_allframes)):
                if len(poses) == 0:
                    continue

                camera = camera.copy()
                camera.world_up = get_up_vector(
                    poses_of_frame=np.array(list(poses.values()), np.float32)[:, :, :3],
                    joint_ids=joint_info.ids)

                gt_boxes = {i: visible_bbox(pose, camera) for i, pose in poses.items()}
                i_people_relevant = [i for i, box in gt_boxes.items() if box[2] > 0]
                gt_boxes_relevant = [gt_boxes[i] for i in i_people_relevant]

                iou_matrix = np.array([[boxlib.iou(gt_box[:4], det[:4])
                                        for det in detections]
                                       for gt_box in gt_boxes_relevant])
                gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

                for i_gt, i_det in zip(gt_indices, det_indices):
                    gt_box = gt_boxes_relevant[i_gt]
                    det_box = detections[i_det]
                    if (iou_matrix[i_gt, i_det] > 0.1 and
                            boxlib.area(det_box) < 2 * boxlib.area(gt_box) and
                            np.max(det_box[2:4]) > 100):

                        i_person = i_people_relevant[i_gt]
                        camcoords = poses[i_person][:, :3]

                        is_occ = poses[i_person][:, 3] == 1
                        is_in_frame = camera.is_visible(camcoords, [1920, 1080])
                        is_valid = np.logical_and(is_in_frame, np.logical_not(is_occ))
                        n_joints_valid = np.count_nonzero(is_valid)
                        if n_joints_valid < 5 or pose_samplers[i_person].should_skip(camcoords):
                            continue

                        ex = ds3d.Pose3DExample(frame, camcoords, bbox=det_box, camera=camera)
                        new_image_replath = f'jta_downscaled/{osp.splitext(relpath)[0]}/' \
                                            f'{i_frame:06d}_{i_person:03d}.jpg'
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_replath),
                            callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    dirname = osp.basename(osp.dirname(video_path))
    if dirname == 'train':
        ds = ds3d.Pose3DDataset(joint_info, train_examples=examples)
    elif dirname == 'val':
        ds = ds3d.Pose3DDataset(joint_info, valid_examples=examples)
    else:
        ds = ds3d.Pose3DDataset(joint_info, test_examples=examples)
    spu.dump_pickle(ds, out_path)


def get_up_vector(poses_of_frame, joint_ids):
    j = joint_ids
    ankle_midpoints = np.mean(poses_of_frame[:, [j.rank, j.lank]], axis=1)
    necks = poses_of_frame[:, j.neck]
    up_vectors = geom3d.unit_vector(necks - ankle_midpoints)
    return geom3d.unit_vector(geom3d.geometric_median(up_vectors, 0.01))


def visible_bbox(pose, camera):
    is_occ = pose[:, 3]
    pose2d = camera.camera_to_image(pose[:, :3])
    is_in_frame = camera.is_visible(pose[:, :3], [1920, 1080])
    is_not_occ = is_occ == 0
    is_valid = np.logical_and(is_not_occ, is_in_frame)
    return boxlib.bb_of_points(pose2d[is_valid])


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('jta.pkl', min_time="2021-12-04T20:56:48")
def make_dataset():
    partial_paths = sorted(glob.glob(f'{DATA_ROOT}/jta_downscaled/tasks/task_result_*.pkl'))
    partial_dss = [spu.load_pickle(p) for p in partial_paths]
    main_ds = partial_dss[0]
    for ds in partial_dss[1:]:
        for i in range(3):
            main_ds.examples[i].extend(ds.examples[i])
    return main_ds


def load_poses(path):
    arr = np.array(spu.load_json(path))
    n_frames = int(np.max(arr[:, 0]))
    result = [collections.defaultdict(lambda: np.empty((22, 4), np.float32)) for _ in
              range(n_frames)]
    for i_frame, i_person, i_joint, x2d, y2d, x3d, y3d, z3d, is_occ, is_self_occ in arr:
        result[int(i_frame - 1)][int(i_person)][int(i_joint)] = np.array(
            [x3d * 1000, y3d * 1000, z3d * 1000, is_occ])
    return result


if __name__ == '__main__':
    main()
