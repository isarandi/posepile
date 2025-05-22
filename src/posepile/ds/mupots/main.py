import os.path as osp

import boxlib
import cameravision
import numpy as np
import scipy.optimize
import simplepyutils as spu

import posepile.datasets3d as ds3d
from posepile import util
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT


@spu.picklecache('mupots-yolo.pkl', min_time="2021-09-16T20:39:52")
def make_mupots():
    joint_names = (
        'htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,spin,head,pelv')
    edges = 'htop-head-neck-spin-pelv-rhip-rkne-rank,neck-rsho-relb-rwri'
    joint_info = JointInfo(joint_names, edges)

    root = f'{DATA_ROOT}/mupots'
    dummy_coords = np.ones((joint_info.n_joints, 3))
    detections_all = spu.load_pickle(f'{root}/yolov3_detections.pkl')
    cameras = load_cameras(f'{root}/camera_intrinsics.json')

    examples_test = []
    for i_seq in range(1, 21):
        annotations = util.load_mat(f'{root}/TS{i_seq}/annot.mat').annotations

        n_frames = annotations.shape[0]
        for i_frame in range(n_frames):
            image_relpath = f'TS{i_seq}/img_{i_frame:06d}.jpg'
            detections_frame = detections_all[image_relpath]
            image_path = f'{root}/{image_relpath}'
            for detection in detections_frame:
                confidence = detection[4]
                if confidence > 0.1:
                    ex = ds3d.Pose3DExample(
                        osp.relpath(image_path, DATA_ROOT),
                        dummy_coords, detection[:4], cameras[i_seq - 1],
                        univ_coords=dummy_coords, scene_name=f'TS{i_seq}')
                    examples_test.append(ex)

    return ds3d.Pose3DDataset(joint_info, test_examples=examples_test)


@spu.picklecache('mupots-yolo-val.pkl', min_time="2021-09-10T00:00:18")
def make_mupots_yolo():
    joint_names = (
        'htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,spin,head,pelv')
    edges = 'htop-head-neck-spin-pelv-rhip-rkne-rank,neck-rsho-relb-rwri'
    joint_info = JointInfo(joint_names, edges)
    order_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 14]

    root = f'{DATA_ROOT}/mupots'
    cameras = load_cameras(f'{root}/camera_intrinsics.json')

    dummy_coords = np.ones((joint_info.n_joints, 3))
    detections_all = spu.load_pickle(f'{root}/yolov3_detections.pkl')

    examples_val = []
    examples_test = []
    for i_seq in range(1, 21):
        annotations = util.load_mat(f'{root}/TS{i_seq}/annot.mat')['annotations']
        camera = cameras[i_seq - 1]

        n_people = annotations.shape[1]
        n_frames = annotations.shape[0]
        for i_frame in range(n_frames):

            image_relpath = f'TS{i_seq}/img_{i_frame:06d}.jpg'
            detections_frame = detections_all[image_relpath]
            image_path = f'{root}/{image_relpath}'
            for detection in detections_frame:
                if detection[4] > 0.1:
                    ex = ds3d.Pose3DExample(
                        image_path, dummy_coords, detection[:4], camera,
                        univ_coords=dummy_coords, scene_name=f'TS{i_seq}')
                    examples_test.append(ex)

            gt_people = []

            for i_person in range(n_people):
                world_coords = np.array(
                    annotations[i_frame, i_person].annot3.T[order_joints], dtype=np.float32)
                univ_world_coords = np.array(
                    annotations[i_frame, i_person].univ_annot3.T[order_joints], dtype=np.float32)
                im_coords = camera.world_to_image(world_coords)
                gt_box = boxlib.expand(boxlib.bb_of_points(im_coords), 1.1)
                ex = ds3d.Pose3DExample(
                    image_path, world_coords, gt_box, camera,
                    univ_coords=univ_world_coords, scene_name=f'TS{i_seq}')
                gt_people.append(ex)

            confident_detections = [det for det in detections_frame if det[-1] > 0.1]
            if confident_detections:
                iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                        for box in confident_detections]
                                       for gt_person in gt_people])
                gt_indices, detection_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
                for i_gt, i_det in zip(gt_indices, detection_indices):
                    if iou_matrix[i_gt, i_det] > 0.1:
                        ex = gt_people[i_gt]
                        ex.bbox = np.array(confident_detections[i_det][:4])
                        examples_val.append(ex)

    return ds3d.Pose3DDataset(
        joint_info, valid_examples=examples_val, test_examples=examples_test)


def load_cameras(json_path):
    json_data = spu.load_json(json_path)
    intrinsic_matrices = [json_data[f'TS{i_seq}'] for i_seq in range(1, 21)]
    return [cameravision.Camera(intrinsic_matrix=intrinsic_matrix, world_up=(0, -1, 0))
            for intrinsic_matrix in intrinsic_matrices]
