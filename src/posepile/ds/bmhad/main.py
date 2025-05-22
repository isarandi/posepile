import argparse
import glob
import itertools
import os
import os.path as osp
import re

import boxlib
import cameralib
import cv2
import ezc3d
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS, logger

import posepile.datasets3d as ds3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example

BMHAD_ROOT = f'{DATA_ROOT}/bmhad'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    all_combinations = list(itertools.product(range(1, 13), range(1, 12), range(1, 6)))
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/bmhad_downscaled/examples/examples_{i_task:06d}.pkl'
    if osp.exists(out_path):
        return

    combinations_per_task = 10
    combinations = all_combinations[
                   i_task * combinations_per_task:(i_task + 1) * combinations_per_task]

    cam_clusters = load_cameras()
    examples = []
    with spu.ThrottledPool() as pool:
        for i_clus, cameras in enumerate(cam_clusters):
            for i_cam, camera in enumerate(cameras):
                for subj, act, rep in combinations:
                    # Missing files in original according to official website
                    if ((subj, act, rep) == (4, 8, 5) or
                            ((subj, act, rep) in [(5, 2, 5), (10, 8, 5)] and i_clus in (2, 3))):
                        continue
                    logger.info(f's{subj:02d}_a{act:02d}_r{rep:02d}')
                    is_kinect = i_clus >= 4
                    i_kinect = i_clus - 4
                    sync_path = (
                        f'{BMHAD_ROOT}/Kinect/Correspondences/corr_moc_kin{i_kinect + 1:02d}'
                        f'_s{subj:02d}_a{act:02d}_r{rep:02d}.txt' if is_kinect else
                        f'{BMHAD_ROOT}/Camera/Correspondences/'
                        f'corr_moc_img_s{subj:02d}_a{act:02d}_r{rep:02d}.txt'
                    )
                    mocap_indices = np.loadtxt(sync_path)[:, 2].astype(int)
                    world_coords_allframes_full = load_from_c3d(
                        f'{BMHAD_ROOT}/Mocap/OpticalData/moc_s{subj:02d}_a{act:02d}_r{rep:02d}.c3d')
                    world_coords_allframes = [world_coords_allframes_full[i] for i in mocap_indices]

                    n_frames = len(world_coords_allframes)
                    sampler = AdaptivePoseSampler2(
                        100, check_validity=True, assume_nan_unchanged=True, buffer_size=n_frames)
                    for i_frame, world_coords in enumerate(
                            spu.progressbar(world_coords_allframes)):
                        if sampler.should_skip(world_coords):
                            continue
                        imcoords = camera.world_to_image(world_coords)
                        box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.05)

                        impath = (
                            f'{BMHAD_ROOT}/Kinect/Kin{i_kinect + 1:02d}/S{subj:02d}/A{act:02d}/'
                            f'R{rep:02d}/kin_k{i_kinect + 1:02d}_s{subj:02d}_a{act:02d}_r{rep:02d}_'
                            f'color_{i_frame:05d}.ppm' if is_kinect else
                            f'{BMHAD_ROOT}/Camera/Cluster{i_clus + 1:02d}/Cam{i_cam + 1:02d}/'
                            f'S{subj:02d}/A{act:02d}/R{rep:02d}/img_l{i_clus + 1:02d}_'
                            f'c{i_cam + 1:02d}_s{subj:02d}_a{act:02d}_r{rep:02d}_{i_frame:05d}.jpg')

                        im_relpath = osp.relpath(impath, BMHAD_ROOT)
                        new_image_relpath = f'bmhad_downscaled/{im_relpath}'
                        new_image_relpath = spu.replace_extension(new_image_relpath, '.jpg')
                        ex = ds3d.Pose3DExample(impath, world_coords, box, camera)
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_relpath),
                            callback=examples.append)

    spu.dump_pickle(examples, out_path)


@spu.picklecache('bmhad.pkl', min_time="2021-12-05T21:47:55")
def make_dataset():
    example_paths = glob.glob(f'{DATA_ROOT}/bmhad_downscaled/examples/examples_*.pkl')
    examples = [ex for p in example_paths for ex in spu.load_pickle(p)]
    examples.sort(key=lambda ex: ex.image_path)

    names = (
        'head,lhead,rhead,rback,backl,backt,lback,lside,bell,chest,rside,lsho1,lsho2,larm,'
        'lelb,lwri,lhan1,lhan2,lhan3,rsho1,rsho2,rarm,relb,rwri,rhan1,rhan2,rhan3,lhipb,'
        'lhipf,lhipl,lleg,lkne,lank,lhee,lfoo,rhipb,rhipf,rhipl,rleg,rkne,rank,rhee,rfoo')
    edges = (
        'head-rhead,lsho1-rsho1-rsho2-rarm-relb-rwri-rhan3,rhan1-rwri-rhan2,'
        'lhipb-rhipb-rhipl-rhipf-rleg-rkne-rank-rhee-rfoo,bell-rside-rhipf,bell-chest'
        'backl-rback-rhipb,backl-backt')
    joint_info = JointInfo(names, edges)
    ds = ds3d.Pose3DDataset(joint_info, examples)
    ds3d.add_masks(
        ds, f'{DATA_ROOT}/bmhad_downscaled/masks',
        relative_root=f'{DATA_ROOT}/bmhad_downscaled')
    return ds


def load_from_c3d(path):
    c3d = ezc3d.c3d(path)
    coords = c3d['data']['points'].transpose(2, 1, 0)[..., :3].astype(np.float32)
    return coords @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32).T


def read_extrinsics(path):
    text = spu.read_file(path)
    pattern = r'Rw=(?P<R>.+) \nTw=(?P<t>.+) \n'
    m = re.match(pattern, text, flags=re.MULTILINE)
    R = np.array(m['R'].split(), np.float32).reshape(3, 3)
    t = np.array(m['t'].split(), np.float32).reshape(3, 1)
    return np.block([[R, t], [np.zeros(3), 1]]).astype(np.float32)


def load_cameras():
    cam_clusters = []
    for cluster_name in ['l01', 'l02', 'l03', 'l04', 'k01', 'k02']:
        cam_cluster = []
        extr = read_extrinsics(f'{BMHAD_ROOT}/Calibration/RwTw_{cluster_name}.txt')
        fs = cv2.FileStorage(
            f'{BMHAD_ROOT}/Calibration/camcfg_{cluster_name}.yml', cv2.FILE_STORAGE_READ)
        n_cameras = int(fs.getNode('Number of cameras').real())
        for i_cam in range(n_cameras):
            cam_node = fs.getNode(f'Camera_{i_cam + 1}')
            intr_mat = cam_node.getNode('K').mat()
            dist_coeffs = np.zeros(5, np.float32)
            dist_coeffs[:4] = cam_node.getNode('Dist').mat()
            posrel = cam_node.getNode('PosRel').mat().reshape(-1)
            t, rotvec = np.split(posrel, [3])
            R = cv2.Rodrigues(rotvec)[0]
            extr_rel = np.block([[R, t[:, np.newaxis]], [np.zeros(3), 1]]).astype(np.float32)
            cam = cameralib.Camera(
                intrinsic_matrix=intr_mat, extrinsic_matrix=extr_rel @ extr,
                distortion_coeffs=dist_coeffs, world_up=(0, 1, 0))
            cam_cluster.append(cam)
        cam_clusters.append(cam_cluster)
    return cam_clusters


if __name__ == '__main__':
    main()
