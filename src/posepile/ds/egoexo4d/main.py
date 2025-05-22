import os.path as osp
import random

import boxlib
import cameralib
from posepile.util import drawing
import cv2
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import scipy.optimize
import simplepyutils as spu
from posepile.paths import DATA_ROOT
from posepile.util import geom3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

DATASET_NAME = 'egoexo4d'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    make_dataset()


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2024-05-06T00:22:24")
def make_dataset():
    examples = {'train': [], 'val': [], 'test': []}

    all_videos = spu.sorted_recursive_glob(f'{DATA_ROOT}/egoexo4d/takes/**/*.mp4')
    needed_highres_videos = [
        v for v in all_videos if 'downscaled' not in v and 'aria' not in v and 'preview' not in v]
    random.shuffle(needed_highres_videos)
    take_data = spu.load_json(f'{DATASET_DIR}/takes.json')
    take_name_to_uid = {item['take_name']: item['take_uid'] for item in take_data}

    take_uid_to_phase = spu.load_json(f'{DATASET_DIR}/annotations/splits.json')['take_uid_to_split']

    joint_info = ds3d.JointInfo(
        'neck,nose,pelv,lsho,lelb,lwri,lhip,lkne,lank,rsho,relb,rwri,rhip,rkne,rank,leye,'
        'lear,reye,rear,'
        'lhan1,lhan2,lhan3,lhan4,lhan5,lhan6,lhan7,lhan8,lhan9,lhan10,lhan11,lhan12,lhan13,'
        'lhan14,lhan15,'
        'lhan16,lhan17,lhan18,lhan19,lhan20,lhan21,rhan1,rhan2,rhan3,rhan4,rhan5,rhan6,'
        'rhan7,rhan8,rhan9,rhan10,rhan11,rhan12,rhan13,rhan14,rhan15,rhan16,rhan17,rhan18,'
        'rhan19,rhan20,rhan21',
        'nose-neck-pelv,rwri-relb-rsho-neck,rank-rkne-rhip-pelv,nose-reye-rear,'
        'rhan1-rhan2-rhan3-rhan4-rhan5,'
        'rhan1-rhan6-rhan7-rhan8-rhan9,'
        'rhan1-rhan10-rhan11-rhan12-rhan13,'
        'rhan1-rhan14-rhan15-rhan16-rhan17,'
        'rhan1-rhan18-rhan19-rhan20-rhan21')
    ji_coco19 = joint_info.select_joints(range(19))
    ji_hand = joint_info.select_joints(range(19, 19 + 42))

    orig_hand_joint_names = [
        f'{side}_{finger}_{i}'
        for side in ['left', 'right']
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i in range(1, 5)]
    orig_hand_joint_names.insert(20, 'right_wrist')
    orig_hand_joint_names.insert(0, 'left_wrist')

    with spu.ThrottledPool() as pool:
        for video_path in spu.progressbar(needed_highres_videos, desc='Processing videos'):
            take_name = spu.split_path(video_path)[-3]
            cam_name = osp.splitext(osp.basename(video_path))[0]
            take_uid = take_name_to_uid[take_name]
            phase = take_uid_to_phase[take_uid]
            if phase == 'test':
                continue

            try:
                body_poses3d = load_gt_poses_as_coco(
                    f'{DATASET_DIR}/annotations/ego_pose/{phase}/body/automatic/{take_uid}.json',
                    ji_coco19)
                hand_poses3d = load_gt_poses_as_coco(
                    f'{DATASET_DIR}/annotations/ego_pose/{phase}/hand/automatic/{take_uid}.json',
                    ji_hand, orig_hand_joint_names)
            except FileNotFoundError:
                print(f'No annotations for {take_name}')
                continue

            cam = load_cameras(
                f'{DATASET_DIR}/annotations/ego_pose/{phase}/camera_pose/{take_uid}.json')[cam_name]


            n_poses = max(len(body_poses3d), len(hand_poses3d))
            body_poses3d = np.concatenate(
                [body_poses3d, np.full((n_poses - len(body_poses3d), 19, 3), np.nan)], axis=0)
            hand_poses3d = np.concatenate(
                [hand_poses3d, np.full((n_poses - len(hand_poses3d), 42, 3), np.nan)], axis=0)
            poses3d = np.concatenate([body_poses3d, hand_poses3d], axis=1) * 1000

            pose_sampler = AdaptivePoseSampler2(
                100, True, True, 200)

            detections_all = spu.load_pickle(
                f'{DATASET_DIR}/detections/takes/'
                f'{take_name}/frame_aligned_videos/downscaled/448/{cam_name}.pkl')
            n_frames = len(detections_all)

            with imageio.get_reader(video_path, 'ffmpeg') as reader:
                for i_frame, frame in enumerate(spu.progressbar(
                        reader, desc=f'{take_name}_{cam_name}', total=n_frames, leave=False)):
                    if i_frame >= len(poses3d):
                        continue

                    detections = detections_all[i_frame] * (2160 / 448)

                    # No detection, perhaps too occluded
                    if len(detections) == 0:
                        # print(f'No detections for {take_name} {cam_name} {i_frame}')
                        continue

                    joints_world = poses3d[i_frame]
                    #print(i_frame, joints_world)

                    # Too few joints labeled
                    if np.sum(geom3d.are_joints_valid(joints_world)) < 6:
                        # print(f'Too few joints for {take_name} {cam_name} {i_frame}')
                        continue

                    # Not enough motion since last kept frame
                    joints_cam = cam.world_to_camera(joints_world)
                    if pose_sampler.should_skip(joints_cam):
                        # print(f'Skipping frame {take_name} {cam_name} {i_frame}')
                        continue

                    # Find detection with best iou overlap
                    joints_im = cam.world_to_image(joints_world)
                    gt_box = boxlib.intersection(
                        boxlib.full(frame.shape),
                        boxlib.expand(boxlib.bb_of_points(joints_im), 1.05))
                    ious = [boxlib.iou(gt_box, det[:4]) for det in detections]

                    # Best detection has too low iou
                    if max(ious) < 0.2:
                        # print(f'Low iou for {take_name} {cam_name} {i_frame}')
                        continue

                    box = detections[np.argmax(ious)][:4]
                    is_in_det = boxlib.contains(box, joints_im)

                    # Too few joints within detection box
                    if np.sum(is_in_det) < 6:
                        # print(f'Too few joints in detection for {take_name} {cam_name} {i_frame}')
                        continue

                    im = frame
                    drawing.draw_box(im, box, color=(0, 255, 0))
                    for i_joint, (x, y) in enumerate(joints_im):
                        color = (0, 0, 255) if is_in_det[i_joint] else (255, 0, 0)
                        if np.all(np.isfinite([x, y])):
                            drawing.circle(im, (x, y), radius=5, color=color, thickness=4)

                    ex = ds3d.Pose3DExample(
                        world_coords=joints_world, image_path=frame, camera=cam, bbox=box)
                    new_image_relpath = (f'{DATASET_NAME}_downscaled/{take_name}/{cam_name}/'
                                         f'{i_frame:06d}.jpg')
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        dict(extreme_perspective=True), callback=examples[phase].append)

    return ds3d.Pose3DDataset(joint_info, examples['train'], examples['val'], examples['test'])


def load_cameras(json_path):
    return {k: load_camera(v)
            for k, v in spu.load_json(json_path).items()
            if k.startswith('cam') or k.startswith('gp')}


def load_camera(v):
    new_intr = np.array(v['camera_intrinsics'], np.float32)
    extr = np.array(v['camera_extrinsics'], np.float32)
    dist_coeffs = np.array(v['distortion_coeffs'], np.float32)
    orig_intr = get_orig_intrinsic_matrix(new_intr, dist_coeffs)
    cam = cameralib.Camera(
        intrinsic_matrix=orig_intr, extrinsic_matrix=extr, distortion_coeffs=dist_coeffs,
        world_up=(0, 0, 1))
    cam.t *= 1000
    return cam


def get_orig_intrinsic_matrix(released_intrinsic_matrix, distortion_coeffs):
    size = (int(released_intrinsic_matrix[0, 2] * 2), int(released_intrinsic_matrix[1, 2] * 2))
    orig_intr = released_intrinsic_matrix.copy()

    def objective(focal):
        orig_intr[0, 0] = focal
        orig_intr[1, 1] = focal
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(orig_intr, distortion_coeffs,
                                                                       size, np.eye(3), balance=0.8)
        return (new_K[0, 0] - released_intrinsic_matrix[0, 0]) ** 2

    optimal_focal = scipy.optimize.minimize_scalar(objective, bounds=(100, 5000), method='bounded',
                                                   options=dict(xatol=1e-4)).x
    orig_intr[0, 0] = optimal_focal
    orig_intr[1, 1] = optimal_focal
    return orig_intr


def load_gt_poses_as_coco(json_path, joint_info, orig_joint_names=None):
    poses = spu.load_json(json_path)
    n_frames = max(int(k) for k in poses.keys()) + 1
    result3d = np.full((n_frames, joint_info.n_joints, 3), np.nan, dtype=np.float32)

    for k, v in poses.items():
        i_frame = int(k)
        if len(v) == 0:
            continue
        if len(v) > 1:
            print(len(v))
            continue

        v = v[0]
        pose = v['annotation3D']
        for jname, jdata in pose.items():
            if joint_info.n_joints == 19:
                i_joint = joint_info.ids[jname.replace('right-', 'r').replace('left-', 'l')[:4]]
            else:
                i_joint = orig_joint_names.index(jname)

            result3d[i_frame, i_joint] = [jdata['x'], jdata['y'], jdata['z']]

    return result3d


if __name__ == '__main__':
    main()
