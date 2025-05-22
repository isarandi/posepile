import glob
import os.path as osp
import re

import boxlib
import cameravision
import numpy as np
import scipy.optimize
import simplepyutils as spu
import yaml

import posepile.datasets2d as ds2d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util import geom3d
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('jrdb2d.pkl', min_time="2022-07-28T15:39:04")
def make_dataset():
    root = f'{DATA_ROOT}/jrdb'
    all_detections = spu.load_pickle(f'{root}/train_dataset/yolov4_detections_distorted.pkl')
    cameras_dist = load_jrdb_cameras(
        f'{root}/train_dataset/calibration/cameras.yaml', undistorted=False)

    anno_paths = glob.glob(f'{root}/train_dataset/labels/labels_2d_pose_coco/*.json')
    dist_dir = f'{root}/train_dataset/images'
    examples = []

    with spu.ThrottledPool() as pool:
        for anno_path in anno_paths:
            anno_content = spu.load_json(anno_path)
            image_id_to_path = {
                x['id']: correct_frame_index(x['file_name']) for x in anno_content['images']}
            annos_per_image = spu.groupby(
                anno_content['annotations'], lambda anno: image_id_to_path[anno['image_id']])
            image_relpaths = sorted(annos_per_image.keys())

            prev_pose_per_track = {}
            for image_relpath in image_relpaths:
                print(image_relpath)
                annos_of_image = annos_per_image[image_relpath]
                image_path = f'{dist_dir}/{image_relpath}'
                image_relpath_to_root = osp.relpath(image_path, DATA_ROOT)
                cam_id = int(re.search('image_(\d)/', image_relpath).group(1))
                camera_dist = cameras_dist[cam_id // 2]
                gt_people = []
                track_ids = []

                for anno in annos_of_image:
                    keypoints = np.array(anno['keypoints']).reshape(17, 3)
                    is_valid = keypoints[:, 2] > 0
                    coords2d = keypoints[:, :2]
                    coords2d[~is_valid] = np.nan
                    imcoords = coords2d
                    bbox = boxlib.expand(boxlib.bb_of_points(imcoords), 1.15)
                    ex = ds2d.Pose2DExample(
                        image_relpath_to_root, imcoords, bbox=bbox, camera=camera_dist)
                    gt_people.append(ex)
                    track_ids.append(anno['track_id'])

                if not gt_people:
                    print(f'No GT people for {image_relpath}')
                    continue

                detections = all_detections[image_relpath]
                iou_matrix = np.array([[boxlib.iou(gt_person.bbox, det[:4])
                                        for det in detections]
                                       for gt_person in gt_people])
                gt_indices, box_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
                for i_gt, i_det in zip(gt_indices, box_indices):
                    if iou_matrix[i_gt, i_det] < 0.1:
                        continue

                    ex = gt_people[i_gt]
                    track_id = track_ids[i_gt]

                    if track_id in prev_pose_per_track:
                        prev_pose = prev_pose_per_track[track_id]
                        prev_is_valid = geom3d.are_joints_valid(prev_pose)
                        now_is_valid = geom3d.are_joints_valid(ex.coords)
                        newly_valid = np.logical_and(np.logical_not(prev_is_valid), now_is_valid)
                        movement_thresh = np.max(ex.bbox[2:]) * 0.1
                        changes = np.linalg.norm(
                            ex.coords[now_is_valid, :2] - prev_pose[now_is_valid, :2], axis=-1)

                        if not np.any(newly_valid) and np.all(changes < movement_thresh):
                            print('small movement')
                            continue

                    prev_pose_per_track[track_id] = ex.coords

                    new_im_relpath = ex.image_path.replace('jrdb', 'jrdb_downscaled')
                    without_ext, ext = osp.splitext(new_im_relpath)
                    new_im_relpath = f'{without_ext}_{track_id:03d}{ext}'
                    ex.bbox = detections[i_det, :4]
                    pool.apply_async(make_efficient_example,
                                     (ex, new_im_relpath), dict(extreme_perspective=True),
                                     callback=examples.append)

    joint_info = JointInfo(
        'htop,reye,leye,rsho,neck,lsho,relb,lelb,pelv,rwri,rhip,lhip,lwri,rkne,lkne,rank,lank',
        'neck-rsho-relb-rwri,htop-neck-pelv-rhip-rkne-rank,htop-reye')
    examples.sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples)


def load_jrdb_cameras(p, undistorted=True):
    with open(p) as f:
        data = yaml.safe_load(f)

    def str_to_mat(s, shape):
        return np.array(list(map(float, s.strip().split()))).reshape(shape)

    cameras = []
    for i in range(5):  # [0, 2, 4, 6, 8]:
        sensorinfo = data['cameras'][f'sensor_{i}']
        R = str_to_mat(sensorinfo['R'], [3, 3])
        K = str_to_mat(sensorinfo['K'], [3, 3])
        D = str_to_mat(sensorinfo['D'], [5])
        T = str_to_mat(sensorinfo['T'], [3])
        cameras.append(cameravision.Camera(
            -R.T @ T, R, intrinsic_matrix=K, world_up=(0, -1, 0),
            distortion_coeffs=D))

    if undistorted:
        for cam in cameras:
            cam.undistort()
            cam.center_principal_point([672, 1053])

    return cameras


def correct_frame_index(impath):
    # The annotations are mismatched with the frames in the released dataset.
    # The image frame index must be increased by 2 to have the correct match.
    i_frame = int(osp.basename(impath).split('.')[0])
    return osp.join(osp.dirname(impath), f'{i_frame + 2:06d}.jpg')


if __name__ == '__main__':
    make_dataset()
