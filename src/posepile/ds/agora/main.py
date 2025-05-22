import os.path as osp
import re

import boxlib
import cameralib
import numpy as np
import rlemasklib
import scipy.optimize
import simplepyutils as spu
import transforms3d
import pandas as pd

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from simplepyutils import logger
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('agora2.pkl', min_time="2021-12-09T21:09:38")
def make_dataset():
    root = f'{DATA_ROOT}/agora'
    all_detections = spu.load_pickle(f'{root}/yolov4_detections.pkl')
    joint_info, i_sel_smplx = get_joint_info()

    def load_examples(section):
        examples = []
        df = pd.read_pickle(f'{root}/SMPL/{section}_withjv.pkl')
        df_smplx = pd.read_pickle(f'{root}/SMPLX/{section}_withjv.pkl')
        with spu.ThrottledPool() as pool:
            for i_row in spu.progressbar(range(len(df))):
                anno = df.iloc[i_row]
                anno_smplx = df_smplx.iloc[i_row]
                image_filename = anno['imgPath']
                image_subdir = section if 'train' in section else 'validation'
                image_path = f'{root}/{image_subdir}/{image_filename}'
                image_relpath = osp.relpath(image_path, DATA_ROOT)
                mask_relpath = image_filename_to_mask_path(image_filename)
                masks = spu.load_pickle(f'{root}/{mask_relpath}')
                detections = all_detections[osp.relpath(image_path, root)]

                gt_people = []
                people_ids = []
                occlusions = []
                for i_person, (is_valid, occlusion, camcoords, camcoords_smplx, modal_mask
                               ) in enumerate(
                    zip(anno['isValid'], anno['occlusion'], anno['gt_joints_3d'],
                        anno_smplx['gt_joints_3d'], masks['modal'])):
                    if not is_valid or occlusion > 75:
                        continue
                    camera = get_camera(
                        camera_location=[anno['camX'], anno['camY'], anno['camZ']],
                        yaw_deg=anno['camYaw'], image_filename=image_filename)

                    world_coords_smpl = camera.camera_to_world(camcoords[:24] * 1000)
                    world_coords_smplx = camera.camera_to_world(camcoords_smplx[i_sel_smplx] * 1000)
                    world_coords = np.concatenate([world_coords_smpl, world_coords_smplx], axis=0)

                    bbox = rlemasklib.to_bbox(modal_mask)
                    ex = ds3d.Pose3DExample(
                        image_relpath, world_coords, bbox, camera, mask=masks['all_people'])
                    gt_people.append(ex)
                    people_ids.append(i_person)
                    occlusions.append(occlusion)

                if not gt_people:
                    logger.info(f'No GT people for {image_filename}')
                    continue

                # Match to detections
                boxes = get_boxes(gt_boxes=[ex.bbox for ex in gt_people], detections=detections)

                # Create crops based on detection boxes
                for ex, box, i_person, occ in zip(gt_people, boxes, people_ids, occlusions):
                    # If there was no detection for this person, and they are more than half
                    # occluded
                    # then let's not create an example for it, it would mislead the training process
                    if np.all(ex.bbox == box) and occ > 50:
                        continue

                    # Skip tiny examples
                    if min(box[2:4]) < 100:
                        continue

                    ex.bbox = box
                    noext = osp.splitext(image_relpath)[0]
                    new_image_relpath = f'{noext}_{i_person}.jpg'.replace(
                        'agora/', 'agora_downscaled/')
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath), callback=examples.append)
        return examples

    examples_train = [ex for i in range(10) for ex in load_examples(f'train_{i}')]
    examples_val = [ex for i in range(10) for ex in load_examples(f'validation_{i}')]
    return ds3d.Pose3DDataset(joint_info, examples_train, examples_val)


def get_joint_info():
    joint_names_smpl = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan')
    edges_smpl = (
        'rhan-rwri-relb-rsho-rcla-thor,head-neck-thor-spin-bell-pelv-rhip-rkne-rank-rtoe')
    joint_info_smpl = JointInfo(joint_names_smpl, edges_smpl)

    i_sel_smpl = list(range(24))
    joint_info_smpl = joint_info_smpl.select_joints(i_sel_smpl)

    joint_names_smplx = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,'
        'lsho,rsho,lelb,relb,lwri,rwri,jaw,leyehf,reyehf,lindex1,lindex2,lindex3,lmiddle1,'
        'lmiddle2,lmiddle3,lpinky1,lpinky2,lpinky3,lring1,lring2,lring3,lthumb1,lthumb2,'
        'lthumb3,rindex1,rindex2,rindex3,rmiddle1,rmiddle2,rmiddle3,rpinky1,rpinky2,rpinky3,'
        'rring1,rring2,rring3,rthumb1,rthumb2,rthumb3,nose,reye,leye,rear,lear,lto2,lto3,lhee,'
        'rto2,rto3,rhee,lthu,lindex,lmiddle,lring,lpinky,rthu,rindex,rmiddle,rring,rpinky,'
        'reyebrow1,reyebrow2,reyebrow3,reyebrow4,reyebrow5,leyebrow5,leyebrow4,leyebrow3,'
        'leyebrow2,leyebrow1,nose1,nose2,nose3,nose4,rnose2,rnose1,nosemiddle,lnose1,lnose2,'
        'reye1,reye2,reye3,reye4,reye5,reye6,leye4,leye3,leye2,leye1,leye6,leye5,rmouth1,rmouth2,'
        'rmouth3,mouthtop,lmouth3,lmouth2,lmouth1,lmouth5,lmouth4,mouthbottom,rmouth4,rmouth5,'
        'rlip1,rlip2,toplip,llip2,llip1,llip3,bottomlip,rlip3')
    edges_smplx = (
        'rwri-relb-rsho-rcla-thor,rank-rhee,rto2-rtoe-rto3,'
        'rear-reye-nose-head-jaw-neck-thor-spin-bell-pelv-rhip-rkne-rank-rtoe')
    joint_info_smplx = JointInfo(joint_names_smplx, edges_smplx)

    i_sel_smplx = [*range(25), 28, 43, *range(55, 67), 68, 71, 73]
    joint_info_smplx = joint_info_smplx.select_joints(i_sel_smplx)
    joint_info_smplx.update_names([n + '_smplx' for n in joint_info_smplx.names])

    joint_names = [*joint_info_smpl.names, *joint_info_smplx.names]
    edges_smplx_renumbered = [
        (i + joint_info_smpl.n_joints, j + joint_info_smpl.n_joints)
        for i, j in joint_info_smplx.stick_figure_edges]
    edges = [*joint_info_smpl.stick_figure_edges, *edges_smplx_renumbered]

    joint_info = JointInfo(joint_names, edges)
    return joint_info, i_sel_smplx


def get_boxes(gt_boxes, detections, iou_thresh=0.5):
    if detections.size == 0:
        return gt_boxes
    iou_matrix = np.array([[boxlib.iou(gt_box[:4], det[:4])
                            for det in detections]
                           for gt_box in gt_boxes])
    gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
    result_boxes = [b for b in gt_boxes]
    for i_gt, i_det in zip(gt_indices, det_indices):
        if iou_matrix[i_gt, i_det] >= iou_thresh:
            result_boxes[i_gt] = detections[i_det][:4]
    return result_boxes


def extract_mask(label_im, i_person, label_colors):
    diff = np.abs(label_im.astype(np.int32) - label_colors[i_person])
    boolmask = np.all(diff < 10, axis=-1)
    return boolmask.astype(np.uint8) * 255


def image_filename_to_mask_path(image_filename):
    pattern = r'(?P<name>(?P<scene>.+?)_(\d+mm_)?(5_10|5_15))(?P<cam>(_cam\d+)?)_(?P<num>\d+)\.png'
    m = re.match(pattern, image_filename)
    if 'trainset' in image_filename:
        phase = 'train'
    elif 'validationset' in image_filename:
        phase = 'validation'
    else:
        assert 'test' in image_filename
        phase = 'test'

    return f'{phase}/{m["scene"]}/{m["name"]}_mask{m["cam"]}_{int(m["num"]):06d}_00000.pkl'


def get_camera(camera_location, yaw_deg, image_filename):
    ground_plane = [0, 0, 0]
    if 'hdri' in image_filename:
        focal_len = 50 / 36 * 3840
        camera_location = [0, 0, 170]
        yaw_deg = 0
        pitch_deg = 0
    elif 'cam00' in image_filename:
        focal_len = 18 / 36 * 3840
        camera_location = [400, -275, 265]
        yaw_deg = 135
        pitch_deg = 30
    elif 'cam01' in image_filename:
        focal_len = 18 / 36 * 3840
        camera_location = [400, 225, 265]
        yaw_deg = -135
        pitch_deg = 30
    elif 'cam02' in image_filename:
        focal_len = 18 / 36 * 3840
        camera_location = [-490, 170, 265]
        yaw_deg = -45
        pitch_deg = 30
    elif 'cam03' in image_filename:
        focal_len = 18 / 36 * 3840
        camera_location = [-490, -275, 265]
        yaw_deg = 45
        pitch_deg = 30
    elif 'ag2' in image_filename:
        focal_len = 28 / 36 * 3840
        camera_location = [0, 0, 170]
        yaw_deg = 0
        pitch_deg = 15
    else:
        ground_plane = [0, -1.7, 0]
        focal_len = 28 / 36 * 3840
        pitch_deg = 0

    intrinsics = np.array([[focal_len, 0, 3840 / 2], [0, focal_len, 2160 / 2], [0, 0, 1]])
    camera_location = np.array(camera_location) / 100
    rot = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
    camera_location = rot @ camera_location + np.array(ground_plane)
    R = transforms3d.euler.euler2mat(-np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0, 'syxz')
    return cameralib.Camera(
        rot_world_to_cam=R, optical_center=camera_location * 1000, intrinsic_matrix=intrinsics,
        world_up=(0, -1, 0))


if __name__ == '__main__':
    make_dataset()
