import glob
import os.path as osp
from collections import defaultdict

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import rlemasklib
import scipy.optimize
import simplepyutils as spu
from posepile.joint_info import JointInfo
import src.posepile.util.drawing as drawing
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from scipy.spatial.transform import Rotation
import more_itertools

DATASET_DIR = f'{DATA_ROOT}/egohumans'


def main():
    make_dataset()

@spu.picklecache('egohumans.pkl', min_time="2023-11-25T13:25:13")
def make_dataset():
    ego_ds = make_ego_dataset()
    exo_ds = make_exo_dataset()
    return ds3d.Pose3DDataset(ego_ds.joint_info, ego_ds.examples[0] + exo_ds.examples[0])


@spu.picklecache('egohumans_ego.pkl', min_time="2023-11-25T13:25:13")
def make_ego_dataset():
    return make_ego_or_exo_dataset(is_ego=True)


@spu.picklecache('egohumans_exo.pkl', min_time="2023-12-01T13:35:22")
def make_exo_dataset():
    return make_ego_or_exo_dataset(is_ego=False)


def make_ego_or_exo_dataset(is_ego=True):
    seq_dirs = sorted(glob.glob(f'{DATASET_DIR}/0*_*/*'))
    examples = []
    detections_all = spu.load_pickle(f'{DATASET_DIR}/yolov4_detections.pkl')
    detections_all = {k.removeprefix('./'): v for k, v in detections_all.items()}
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear')

    with spu.ThrottledPool() as pool:
        for pbar, seq_dir in spu.zip_progressbar(seq_dirs):
            seq_id = spu.path_range(seq_dir, -2, None)
            pbar.set_description(seq_id)

            cameras = load_cameras(seq_id)

            for camname, camera_per_frame in cameras.items():
                if is_ego != ('aria' in camname):
                    continue

                pose_sampler = AdaptivePoseSampler2(
                    100, True, True, 200)

                maybe_rgb = '/rgb' if 'aria' in camname else ''
                ego_or_exo = 'ego' if 'aria' in camname else 'exo'

                for i_frame, camera in enumerate(spu.progressbar(
                        camera_per_frame, leave=False, desc=camname)):
                    try:
                        poses3d_ = np.load(
                            f'{seq_dir}/processed_data/poses3d/{i_frame + 1:05d}.npy',
                            allow_pickle=True).item()
                    except FileNotFoundError:
                        print(f'Not found: {seq_dir}/processed_data/poses3d/{i_frame + 1:05d}.npy')
                        continue

                    human_names = [n for n in poses3d_.keys() if n != camname]
                    poses3d = np.array(
                        [poses3d_[n][:, :3] for n in human_names], np.float32) * 1000
                    poses2d = camera.world_to_image(poses3d)
                    poses3d_cam = camera.world_to_camera(poses3d)

                    if is_ego:
                        is_valid = np.logical_and(
                            np.linalg.norm(poses2d - [704, 704], axis=-1) < 500,
                            poses3d_cam[..., 2] > 0)
                        poses3d[~is_valid] = np.nan
                        poses2d[~is_valid] = np.nan

                    gt_boxes = np.array([boxlib.expand(boxlib.bb_of_points(p), 1.05)
                                         for p in poses2d])

                    image_path = (
                        f'{seq_dir}/{ego_or_exo}/{camname}/images{maybe_rgb}/{i_frame + 1:05d}.jpg')
                    image_relpath = osp.relpath(image_path, DATASET_DIR)
                    try:
                        detections = detections_all[image_relpath]
                    except KeyError:
                        assert not osp.exists(image_path)
                        print(f'No image for {image_relpath}, continuing...')
                        continue

                    iou_matrix = np.array([[boxlib.iou(gt_box, det[:4])
                                            for det in detections]
                                           for gt_box in gt_boxes])
                    gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
                    im = None

                    for i_person, i_det in zip(gt_indices, det_indices):
                        if iou_matrix[i_person, i_det] < 0.2:
                            continue
                        if np.any(np.isnan(poses2d[i_person])):
                            continue

                        is_in_det = boxlib.contains(detections[i_det], poses2d[i_person])
                        if np.sum(is_in_det) < 6:
                            continue

                        box = detections[i_det][:4]
                        # drawing.draw_box(im, detections[i_det], color=(0, 255, 0))
                        # for i_joint, (x, y) in enumerate(poses2d[i_person]):
                        #     color = (0, 0, 255) if is_in_det[i_joint] else (255, 0, 0)
                        #     drawing.circle(im, (x, y), radius=5, color=color)

                        if pose_sampler.should_skip(poses3d_cam[i_person]):
                            continue

                        if im is None:
                            im = imageio.imread(image_path)

                        ex = ds3d.Pose3DExample(
                            image_path=im,
                            #image_path=f'egohumans/{image_relpath}',
                            camera=camera,
                            bbox=box, world_coords=poses3d[i_person])
                        relpath_noext = osp.splitext(image_relpath)[0]
                        new_image_relpath = (
                            f'egohumans_downscaled/{relpath_noext}_{i_person:02d}.jpg')

                        pool.apply_async(
                            make_efficient_example, (ex, new_image_relpath),
                            kwargs=dict(extreme_perspective=True),
                            callback=examples.append)

    return ds3d.Pose3DDataset(joint_info, examples)


def load_cameras(seq):
    seq_dir = f'{DATASET_DIR}/{seq}'
    # Collect intrinsics and distortion coefficients
    intrinsic_lines = spu.read_lines(f'{seq_dir}/colmap/workplace/cameras.txt')[3:]
    intrinsic_matrices = []
    distortion_coeffs = []

    for line in intrinsic_lines:
        parts = line.split()
        fx, fy = parts[4:6]
        cx, cy = parts[6:8]
        k1, k2, k3, k4 = parts[8:12]
        intrinsic_matrices.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32))
        distortion_coeffs.append(np.array([k1, k2, k3, k4], np.float32))

    cameras = defaultdict(list)
    colmap_from_aria_transforms = spu.load_pickle(
        f'{seq_dir}/colmap/workplace/colmap_from_aria_transforms.pkl')
    num_ego = len(colmap_from_aria_transforms)

    # Ego cameras
    for i in range(num_ego):
        postmultiply_extr = (
                np.linalg.inv(colmap_from_aria_transforms[f'aria{i + 1:02d}']) @
                colmap_from_aria_transforms['aria01'])
        calib_paths = spu.sorted_recursive_glob(
            f'{DATASET_DIR}/{seq}/ego/aria{i + 1:02d}/calib/*.txt')
        for calib_path in calib_paths:
            calib = spu.read_lines(calib_path)
            extr = (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float32) @
                    np.array(calib[3].strip().split(' ')).astype(np.float32).reshape((4, 3)).T @
                    postmultiply_extr)
            cameras[f'aria{i + 1:02d}'].append(
                cameralib.Camera(
                    intrinsic_matrix=intrinsic_matrices[i],
                    distortion_coeffs=distortion_coeffs[i],
                    extrinsic_matrix=extr,
                    world_up=(0, 0, 1)))

    # Exo cameras
    determinant = np.linalg.det(colmap_from_aria_transforms['aria01'][:3, :3])
    normalized_colmap_from_aria = colmap_from_aria_transforms['aria01'] / determinant ** (1 / 3)
    cam_lines = spu.read_lines(f'{seq_dir}/colmap/workplace/images.txt')[4::2]

    exo_cameras = defaultdict(dict)
    for line in cam_lines:
        parts = line.split()
        filename = parts[9]
        if 'aria' in filename:
            continue

        qw, qx, qy, qz = parts[1:5]
        rotmat = Rotation.from_quat([qx, qy, qz, qw]).as_matrix().astype(np.float32)
        trans = np.array(parts[5:8], np.float32)
        extr_3x4 = np.concatenate([rotmat, trans[:, np.newaxis]],
                                  axis=1) @ normalized_colmap_from_aria

        camname = filename.split('/')[0]
        try:
            cam_num = int(camname[-2:])
        except ValueError:
            continue

        frame_id = int(filename.split('/')[1].split('.')[0])
        exo_cameras[camname][frame_id] = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrices[cam_num + num_ego - 1],
            distortion_coeffs=distortion_coeffs[cam_num + num_ego - 1],
            extrinsic_matrix=extr_3x4,
            world_up=(0, 0, 1))

    n_frames = len(cameras['aria01'])
    for camname in exo_cameras:
        if osp.isfile(f'{DATA_ROOT}/egohumans/{seq}/colmap/workplace/{camname}.npy'):
            extr = np.load(f'{DATA_ROOT}/egohumans/{seq}/colmap/workplace/{camname}.npy')
            cam_num = int(camname[-2:])
            cameras[camname] = [cameralib.Camera(
                intrinsic_matrix=intrinsic_matrices[cam_num + num_ego - 1],
                distortion_coeffs=distortion_coeffs[cam_num + num_ego - 1],
                extrinsic_matrix=extr,
                world_up=(0, 0, 1)) for _ in range(n_frames)]
            continue

        frame_ids_with_extr = sorted(exo_cameras[camname].keys())
        assert frame_ids_with_extr[0] == 1
        for prev_frame_id, target_frame_id in more_itertools.pairwise(frame_ids_with_extr):
            if prev_frame_id >= n_frames:
                break

            gap_size = min(target_frame_id, n_frames) - prev_frame_id
            cameras[camname] += [exo_cameras[camname][prev_frame_id].copy()
                                 for _ in range(gap_size // 2)]
            cameras[camname] += [exo_cameras[camname][target_frame_id].copy()
                                 for _ in range(gap_size - gap_size // 2)]

        cameras[camname] += [exo_cameras[camname][frame_ids_with_extr[-1]].copy()
                             for _ in range(n_frames - len(cameras[camname]))]

        for cam in cameras[camname]:
            assert np.all(np.isfinite(cam.t))

    for camname, cams in cameras.items():
        for cam in cams:
            cam.t *= 1000
    return cameras


if __name__ == '__main__':
    main()
