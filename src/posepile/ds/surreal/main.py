import argparse
import functools
import glob

import cameralib
import numpy as np
import rlemasklib
import scipy.optimize
import scipy.spatial.distance
import simplepyutils as spu
import transforms3d
from simplepyutils import FLAGS
from smplfitter.np import get_cached_body_model

import posepile.datasets3d as ds3d
from posepile import util
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-vertices', action=spu.argparse.BoolAction)
    spu.initialize(parser)
    make_dataset(store_vertices=FLAGS.store_vertices,
                 adaptive_threshold=500 if FLAGS.store_vertices else 100)


selected_joints = [*range(1, 24), 0]
selected_mesh_points = [*selected_joints, *range(24, 24 + 6890)]


@spu.picklecache('surreal_adv.pkl', min_time="2022-08-05T21:42:49")
def make_dataset(store_vertices=False, adaptive_threshold=100):
    joint_info = get_smpl_mesh_joint_info() if store_vertices else get_smpl_joint_info()
    matpaths = [
        *glob.glob(f'{DATA_ROOT}/surreal/train/*/*/*_info.mat'),
        *glob.glob(f'{DATA_ROOT}/surreal/val/*/*/*_info.mat'),
        *glob.glob(f'{DATA_ROOT}/surreal/test/*/*/*_info.mat')]

    examples_train = []
    examples_val = []
    examples_test = []
    with spu.ThrottledPool() as pool:
        for i, matpath in enumerate(spu.progressbar(matpaths)):
            if 'surreal/train' in matpath:
                container = examples_train
            elif 'surreal/val' in matpath:
                container = examples_val
            else:
                container = examples_test

            pool.apply_async(
                process_matfile, (matpath, store_vertices, adaptive_threshold),
                callback=container.extend)

    examples_train.sort(key=lambda ex: ex.image_path)
    examples_val.sort(key=lambda ex: ex.image_path)
    examples_test.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, examples_train, examples_val, examples_test)


def get_smpl_joint_info():
    body_joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    joint_names = [body_joint_names[j] for j in selected_joints]
    edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe'
    return JointInfo(joint_names, edges)


def get_smpl_mesh_joint_info():
    try:
        mesh_joint_info = spu.load_pickle(
            f'{DATA_ROOT}/skeleton_conversion/smpl_mesh_joint_info.pkl')
    except FileNotFoundError:
        mesh_joint_info = save_mesh_joint_info()

    joint_info = get_smpl_joint_info()
    return JointInfo(joint_info.names + mesh_joint_info.names, joint_info.stick_figure_edges)


@functools.lru_cache
def load_smpl_model(gender):
    return SMPLBodyModel('smpl', gender=gender)


def process_matfile(matpath, store_vertices=False, adaptive_threshold=100):
    matdata = util.load_mat(matpath)
    segpath = matpath.replace('_info.mat', '_segm.mat')
    seg_data = util.load_mat(segpath)

    if matdata['joints2D'].ndim != 3:
        print(f'Could not process {matpath}')
        return []

    camera_location = np.array(matdata['camLoc']) * 1000
    released_coords3d = matdata['joints3D'].transpose(2, 1, 0) * 1000
    # 159 instead of 160 because after mirroring the images, things get shifted by one pixel
    intrinsics = np.array([[600, 0, 159], [0, 600, 120], [0, 0, 1]])

    camera = cameralib.Camera(
        extrinsic_matrix=get_extrinsic(camera_location),
        intrinsic_matrix=intrinsics, world_up=(0, -1, 0))
    exs = []
    image_dir = matpath.replace('_info.mat', '')
    n_frames = len(matdata['pose'].T)
    pose_sampler = AdaptivePoseSampler(adaptive_threshold)

    gender = 'f' if matdata['gender'][0] == 0 else 'm'
    root_male = np.array([-0.00217368, -0.24078917, 0.02858379]) * 1000
    root_female = np.array([-0.00087631, -0.21141872, 0.02782112]) * 1000
    root_pos = root_male if gender == 'm' else root_female
    body_model = load_smpl_model(gender)

    zrot = np.array(matdata['zrot'], np.float32)
    computed_coords3d = get_smpl_joints(
        body_model, matdata['pose'].T, matdata['shape'].T, zrot, store_vertices=store_vertices)

    selected_points = selected_mesh_points if store_vertices else selected_joints

    for i_frame in range(n_frames):
        image_path = f'{image_dir}/frame_{i_frame:06d}.jpg'
        # should be trans = released_coords3d[i_frame][0] - computed_coords3d[i_frame][0]
        # TODO fix. Left as is, for consistency with how training was performed for SarandiWACV23
        trans = released_coords3d[i_frame][0] - root_pos
        world_coords = (computed_coords3d[i_frame] + trans)[selected_points]

        if pose_sampler.should_skip(world_coords):
            continue

        mask_encoded = rlemasklib.encode(seg_data[f'segm_{i_frame + 1}'][:, ::-1] > 0)
        bbox = rlemasklib.to_bbox(mask_encoded)

        if np.min(bbox[2:]) == 0:
            continue
        ex = ds3d.Pose3DExample(
            image_path, world_coords, bbox=bbox, camera=camera, mask=mask_encoded)
        exs.append(ex)

    return exs


def get_extrinsic(T):
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).T
    T_world2bcam = -1 * np.dot(R_world2bcam, T)
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam
    RT = np.concatenate([R_world2cv, T_world2cv[:, np.newaxis]], axis=1)
    RT[0] *= -1
    return RT


def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec, axis=-1, keepdims=True)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = RzBody @ Rpelvis
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(R90 @ globRotMat)
    globRotVec = globRotAx * globRotAngle
    return globRotVec


def get_smpl_joints(body_model, pose_params, shape_params, zrot, store_vertices=False):
    zero = np.zeros_like(zrot)
    one = np.ones_like(zrot)
    RzBody = np.array(((np.cos(zrot), -np.sin(zrot), zero),
                       (np.sin(zrot), np.cos(zrot), zero),
                       (zero, zero, one))).transpose(2, 0, 1)

    for i in range(len(pose_params)):
        pose_params[i, :3] = rotateBody(RzBody[i], pose_params[i, :3])

    result = body_model(pose_params, shape_params, return_vertices=store_vertices)
    points = (np.concatenate([result['joints'], result['vertices']], axis=1)
              if store_vertices else result['joints'])
    return points * 1000


def make_mesh_joint_info(body_model):
    verts = body_model.single()['vertices']
    verts_mirror = verts.copy()
    verts_mirror[:, 0] *= -1

    dist = scipy.spatial.distance.cdist(verts_mirror, verts)
    vert_indices, mirror_indices = scipy.optimize.linear_sum_assignment(dist)
    i_centrals = np.argwhere(mirror_indices == vert_indices)[:, 0]
    i_lefts = np.argwhere((verts - verts[mirror_indices])[:, 0] > 0)[:, 0]

    names = [None] * body_model.num_vertices
    for i_within_centrals, i_joint in enumerate(i_centrals):
        names[i_joint] = f'c{i_within_centrals:04d}'

    for i_within_lefts, i_joint in enumerate(i_lefts):
        names[i_joint] = f'l{i_within_lefts:04d}'
        names[mirror_indices[i_joint]] = f'r{i_within_lefts:04d}'

    return JointInfo(names, [])


def save_mesh_joint_info():
    body_model = get_cached_body_model('smpl', 'neutral')
    joint_info = make_mesh_joint_info(body_model)
    spu.dump_pickle(joint_info, f'{DATA_ROOT}/skeleton_conversion/smpl_mesh_joint_info.pkl')
    return joint_info


if __name__ == '__main__':
    main()
