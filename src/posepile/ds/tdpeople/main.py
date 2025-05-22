import argparse
import glob
import itertools
import multiprocessing
import os.path as osp

import cameralib
import cv2
import imageio.v2 as imageio
import numpy as np
import posepile.compositing
import posepile.datasets3d as ds3d
import rlemasklib
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--composite', action=spu.argparse.BoolAction)
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.composite:
        if FLAGS.stage == 1:
            make_stage1()
        elif FLAGS.stage == 2:
            make_composited_dataset()
    else:
        make_dataset()


@spu.picklecache('tdpeople.pkl', min_time="2021-12-08T20:28:55")
def make_dataset():
    joint_selection = [*range(14), *range(33, 37), 53, *range(57, 67)]
    joint_names = (
        'pelv,spin,spi1,spi2,neck,head,reye,leye,htop,lcla,lsho,lelb,lwri,lhan,rcla,rsho,relb,rwri,'
        'rhan,lhip,lkne,lank,lfoo,ltoe,rhip,rkne,rank,rfoo,rtoe')
    edges = (
        'pelv-spin-spi1-spi2-neck-head-htop,pelv-rhip-rkne-rank-rfoo-rtoe,'
        'spi2-rcla-rsho-relb-rwri-rhan,leye-reye')

    joint_info = JointInfo(joint_names, edges)
    image_paths = sorted(glob.glob(f'{DATA_ROOT}/3dpeople/rgb/**/*.jpg', recursive=True))
    pool = multiprocessing.Pool()
    all_coords = list(spu.progressbar(
        pool.imap(get_coords_for_image_path, image_paths), total=len(image_paths)))

    useful_image_paths = []
    useful_coords = []
    pose_sampler = AdaptivePoseSampler(100)
    for image_path, coords in zip(spu.progressbar(image_paths), all_coords):
        world_coords = coords[:, -3:] * 100
        if pose_sampler.should_skip(world_coords):
            continue
        useful_image_paths.append(image_path)
        useful_coords.append(coords)

    examples = list(spu.progressbar(
        pool.imap(
            create_example,
            zip(useful_image_paths, useful_coords, itertools.repeat(joint_selection))),
        total=len(useful_image_paths)))
    return ds3d.Pose3DDataset(joint_info, examples)


def get_coords_for_image_path(image_path):
    parts = spu.split_path(image_path)
    lastdir_removed = '/'.join(parts[:-2] + parts[-1:])
    skeleton_path = lastdir_removed.replace('/rgb/', '/skeleton/').replace('.jpg', '.txt')
    return np.loadtxt(skeleton_path, dtype=np.float32)


def create_example(args):
    image_path, coords, joint_selection = args
    parts = spu.split_path(image_path)
    lastdir_removed = '/'.join(parts[:-2] + parts[-1:])
    segmentation_path = lastdir_removed.replace(
        '/rgb/', '/segmentation_clothes/').replace('.jpg', '.png')
    mask = load_mask(segmentation_path)
    bbox = rlemasklib.to_bbox(mask)
    coords2d = coords[:, :2]
    world_coords = coords[:, -3:] * 100
    camera = calibrate_camera(coords2d, world_coords)
    image_relpath = osp.relpath(image_path, DATA_ROOT)
    return ds3d.Pose3DExample(
        image_relpath, world_coords[joint_selection], bbox, camera, mask=mask)


@spu.picklecache('muco_tdpeople_stage1.pkl', min_time="2021-12-08T20:17:53")
def make_stage1():
    ds = make_dataset()
    print(len(ds.examples[0]))
    ds.examples[0] = posepile.compositing.make_combinations(
        ds.examples[0], n_count=len(ds.examples[0]) // 3, rng=np.random.RandomState(0),
        n_people_per_image=4, output_dir=f'{DATA_ROOT}/muco_3dpeople/images/',
        imshape=(480, 640))
    return ds


@spu.picklecache('muco_tdpeople.pkl', min_time="2020-07-18T00:43:46")
def make_composited_dataset():
    ds = make_stage1()
    detections = spu.load_pickle(f'{DATA_ROOT}/muco_3dpeople/yolov4_detections.pkl')
    return posepile.compositing.make_composited_dataset(ds, detections)


def load_mask(segmentation_path):
    mask = imageio.imread(segmentation_path)[..., :3]
    mask = np.any(np.abs(mask.astype(np.int32) - 153) > 10, axis=-1)
    return rlemasklib.encode(mask)


def calibrate_camera(coords2d, coords3d):
    flags = (
            cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_FOCAL_LENGTH |
            cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |
            cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
            cv2.CALIB_FIX_TANGENT_DIST)

    mean3d = np.mean(coords3d, axis=0, keepdims=True)
    coords3d = coords3d - mean3d
    intrinsic_matrix = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], np.float32)
    reproj_error, intr, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        coords3d[np.newaxis].copy(), coords2d[np.newaxis].copy(), cameraMatrix=intrinsic_matrix,
        imageSize=(640, 480),
        distCoeffs=None, flags=flags)
    rot_matrix = cv2.Rodrigues(rvecs[0])[0]
    t = tvecs[0] - rot_matrix @ mean3d.reshape(3, 1)
    extrinsic_matrix = np.concatenate([rot_matrix, t], axis=1)
    return cameralib.Camera(intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix)


if __name__ == '__main__':
    main()
