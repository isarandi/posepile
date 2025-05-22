import argparse
import glob
import os.path as osp
import queue
import threading

import barecat
import cameralib
import cv2
import numpy as np
import posepile.datasets3d as ds3d
import posepile.ds.surreal.main as surreal_main
import rlemasklib
import simplepyutils as spu
from posepile import util
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler


def main():
    parser = argparse.ArgumentParser()
    spu.initialize(parser)
    save_dataset(100)


def save_dataset(adaptive_threshold=100):
    matpaths = [
        *glob.glob(f'{DATA_ROOT}/surreal/train/*/*/*_info.mat'),
        *glob.glob(f'{DATA_ROOT}/surreal/val/*/*/*_info.mat'),
        *glob.glob(f'{DATA_ROOT}/surreal/test/*/*/*_info.mat')]

    q = queue.Queue(32)
    writer_thread = threading.Thread(target=barecat_writer_thread_main, args=(q,))
    writer_thread.start()

    with spu.ThrottledPool() as pool:
        for i, matpath in enumerate(spu.progressbar(matpaths)):
            pool.apply_async(
                process_matfile, (matpath, adaptive_threshold), callback=q.put)
    q.put(None)
    writer_thread.join()


def barecat_writer_thread_main(q):
    joint_info = surreal_main.get_smpl_joint_info()
    with barecat.Barecat(
            f'{DATA_ROOT}/bc/surreal.barecat', overwrite=True, readonly=False,
            auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=joint_info.names,
            joint_edges=joint_info.stick_figure_edges,
            train_bone_lengths=np.array([], np.float32),
            trainval_bone_lengths=np.array([], np.float32))

        while (exs := q.get()) is not None:
            for ex in exs:
                if 'surreal/train' in ex.image_path:
                    phase_name = 'train'
                elif 'surreal/val' in ex.image_path:
                    phase_name = 'val'
                else:
                    phase_name = 'test'

                bc_writer[f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


def example_to_dict(ex):
    result = dict(
        impath=ex.image_path,
        bbox=np.round(ex.bbox).astype(np.int16),
        parameters=ex.parameters,
        cam=dict(
            rotvec_w2c=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            loc=ex.camera.t,
            intr=ex.camera.intrinsic_matrix[:2],
            up=ex.camera.world_up
        )
    )
    if (ex.camera.distortion_coeffs is not None and
            np.count_nonzero(ex.camera.distortion_coeffs) > 0):
        result['cam']['distcoef'] = ex.camera.distortion_coeffs

    if ex.mask is not None:
        result['mask'] = rlemasklib.compress(ex.mask, zlevel=-1)
    return result


def process_matfile(matpath, adaptive_threshold=100):
    matdata = util.load_mat(matpath)
    segpath = matpath.replace('_info.mat', '_segm.mat')
    seg_data = util.load_mat(segpath)

    if matdata['joints2D'].ndim != 3:
        print(f'Could not process {matpath}')
        return []

    camera_location = np.array(matdata['camLoc']) * 1000
    released_coords3d = matdata['joints3D'].transpose(2, 1, 0) * 1000
    # 159 instead of 160 because after mirroring the images, things get shifted by one pixel
    intrinsics = np.array([[600, 0, 159.5], [0, 600, 119.5], [0, 0, 1]])

    camera = cameralib.Camera(
        extrinsic_matrix=surreal_main.get_extrinsic(camera_location),
        intrinsic_matrix=intrinsics, world_up=(0, -1, 0))
    exs = []
    image_dir = osp.relpath(matpath.replace('_info.mat', ''), DATA_ROOT)
    n_frames = len(matdata['pose'].T)
    pose_sampler = AdaptivePoseSampler(adaptive_threshold)

    gender = 'f' if matdata['gender'][0] == 0 else 'm'
    zrot = np.array(matdata['zrot'], np.float32)
    transformed_pose = transform_pose(matdata['pose'].T, zrot)
    shape = matdata['shape'].T

    body_model = surreal_main.load_smpl_model(gender)
    computed_coords3d = get_smpl_joints(body_model, transformed_pose, shape)

    for i_frame in range(n_frames):
        trans = released_coords3d[i_frame][0] - computed_coords3d[i_frame][0]
        world_coords = computed_coords3d[i_frame] + trans
        if pose_sampler.should_skip(world_coords):
            continue

        image_path = f'{image_dir}/frame_{i_frame:06d}.jpg'
        mask_encoded = rlemasklib.encode(seg_data[f'segm_{i_frame + 1}'][:, ::-1] > 0)
        bbox = rlemasklib.to_bbox(mask_encoded)

        if np.min(bbox[2:]) == 0:
            continue
        ex = ds3d.Pose3DExample(
            image_path, world_coords=None, bbox=bbox, camera=camera, mask=mask_encoded,
            parameters=dict(
                type='smpl', gender=gender, pose=transformed_pose[i_frame], shape=shape[i_frame],
                trans=trans))
        exs.append(ex)

    return exs


def transform_pose(pose_params, zrot):
    zero = np.zeros_like(zrot)
    one = np.ones_like(zrot)
    RzBody = np.array(((np.cos(zrot), -np.sin(zrot), zero),
                       (np.sin(zrot), np.cos(zrot), zero),
                       (zero, zero, one))).transpose(2, 0, 1)
    pose_params = pose_params.copy()
    for i in range(len(pose_params)):
        pose_params[i, :3] = surreal_main.rotateBody(RzBody[i], pose_params[i, :3])
    return pose_params


def get_smpl_joints(body_model, pose_params, shape_params):
    result = body_model(pose_params, shape_params, return_vertices=False)
    return result['joints'] * 1000


if __name__ == '__main__':
    main()
