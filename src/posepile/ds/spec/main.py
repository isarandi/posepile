import collections
import os.path as osp

import boxlib
import cameravision
import cv2
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.ds.agora.main import get_boxes
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import logger
from smpl.numpy import SMPL


@spu.picklecache('spec2.pkl', min_time="2022-08-05T16:57:34")
def make_dataset():
    root = f'{DATA_ROOT}/spec/spec-syn'
    all_detections = spu.load_pickle(f'{root}/yolov4_detections.pkl')

    anno_train = np.load(f'{root}/annotations/train.npz')
    anno_test = np.load(f'{root}/annotations/test.npz')

    joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe'
    joint_info = JointInfo(joint_names, edges)
    body_model = SMPL(model_root=f'{DATA_ROOT}/body_models/smpl')


    openpose_joint_names = (
        'head,neck,rsho,relb,rwri,lsho,lelb,lwri,pelv,rhip,rkne,rank,lhip,lkne,lank,reye,leye,'
        'rear,lear,lto2,lto3,lhee,rto2,rto3,rhee').split(',')
    commons = 'neck,rsho,relb,rwri,lsho,lelb,lwri,pelv,rhip,rkne,rank,lhip,lkne,lank'.split(',')
    S_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    i_3d = [joint_info.ids[n] for n in commons]
    i_2d = [openpose_joint_names.index(n) for n in commons]
    S_selector = [joint_info.ids[n] for n in S_names]

    def load_examples(a):
        examples = []

        groups = collections.defaultdict(list)
        for i, imgname in enumerate(a['imgname']):
            groups[imgname].append(i)

        with spu.ThrottledPool() as pool:
            for imgname, ids in groups.items():
                gt_people = []
                pose_unks = get_smpl_joints(
                    body_model, np.array([a['pose'][i_pers] for i_pers in ids]),
                    np.array([a['shape'][i_pers] for i_pers in ids]))

                for i_pers, pose_unk in zip(ids, pose_unks):
                    openpose_gt = a['openpose_gt'][i_pers]
                    cam_rotmat = a['cam_rotmat'][i_pers]
                    cam_int = a['cam_int'][i_pers]
                    S = a['S'][i_pers, :, :3] * 1000

                    pose2d = openpose_gt[..., :2]
                    cam = cameravision.Camera(
                        rot_world_to_cam=cam_rotmat, intrinsic_matrix=cam_int, world_up=(0, -1, 0))
                    rot, trans = calibrate_extrinsics(pose2d[i_2d], pose_unk[i_3d], cam_int)
                    scale_factor = get_scale(S) / get_scale(pose_unk[S_selector])
                    camcoords = (pose_unk @ rot.T + trans) * scale_factor
                    world_coords = cam.camera_to_world(camcoords)
                    imcoords = cam.world_to_image(world_coords)
                    posebox = boxlib.expand(boxlib.bb_of_points(imcoords), 1.15)
                    image_relpath = f'spec/spec-syn/{imgname}'
                    ex = ds3d.Pose3DExample(image_relpath, world_coords, posebox, cam)
                    gt_people.append(ex)

                if not gt_people:
                    logger.info(f'No GT people for {imgname}')
                    continue

                detections = all_detections[imgname]
                # Match to detections
                boxes = get_boxes(
                    gt_boxes=[ex.bbox for ex in gt_people], detections=detections, iou_thresh=0.3)

                # Create crops based on detection boxes
                for ex, box, i_pers in zip(gt_people, boxes, ids):
                    # Skip undetected
                    if np.all(ex.bbox == box):
                        continue

                    # Skip tiny examples
                    if min(box[2:4]) < 100:
                        continue

                    ex.bbox = box
                    noext = osp.splitext(image_relpath)[0]
                    subdir = osp.basename(noext).split('_')[3]
                    new_image_relpath = f'{noext}_{i_pers:06d}.jpg'.replace(
                        'spec/', 'spec_downscaled/')

                    d = osp.dirname(new_image_relpath)
                    b = osp.basename(new_image_relpath)
                    new_image_relpath = f'{d}/{subdir}/{b}'
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        dict(extreme_perspective=True), callback=examples.append)

        return examples

    examples_train = load_examples(anno_train)
    examples_test = load_examples(anno_test)
    return ds3d.Pose3DDataset(joint_info, examples_train, test_examples=examples_test)


def get_smpl_joints(body_model, pose_params, shape_params):
    return body_model(pose_params, shape_params, return_vertices=False)['joints'] * 1000

def get_scale(poses):
    mean = np.mean(poses, axis=-2, keepdims=True)
    return np.sqrt(np.mean(np.square(poses - mean), axis=(-2, -1), keepdims=False))



def calibrate_extrinsics(image_coords2d, world_coords3d, intrinsic_matrix):
    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |
             cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
             cv2.CALIB_FIX_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH)
    coords2d = image_coords2d[np.newaxis].astype(np.float32)
    coords3d = world_coords3d[np.newaxis].astype(np.float32)
    reproj_error, intrinsic_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        coords3d, coords2d, cameraMatrix=intrinsic_matrix, imageSize=(1920, 1080),
        distCoeffs=None, flags=flags)
    rot_matrix = cv2.Rodrigues(rvecs[0])[0]
    t = tvecs[0][:, 0]
    return rot_matrix, t


if __name__ == '__main__':
    make_dataset()
