import argparse
import glob
import os
import os.path as osp

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
from simplepyutils import logger
from smpl.numpy import SMPL

import posepile.datasets3d as ds3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


def main():
    parser = argparse.ArgumentParser()
    spu.initialize(parser)
    make_dataset()


@spu.picklecache('behave.pkl', min_time="2020-09-13T20:00:00")
def make_dataset():
    root = f'{DATA_ROOT}/behave'
    joint_info, smpl_reordering, openpose_reordering = make_joint_info()
    all_detections = spu.load_pickle(f'{root}/yolov4_detections.pkl')

    all_detections = {k: [np.array(x[:4]) for x in v] for k, v in all_detections.items()}
    examples_train = []
    examples_test = []
    smpl_male = SMPL(model_root=f'{DATA_ROOT}/body_models/smpl', gender='m')
    smpl_female = SMPL(model_root=f'{DATA_ROOT}/body_models/smpl', gender='f')
    split = spu.load_json(f'{root}/split.json')

    with spu.ThrottledPool() as pool:
        seq_names = os.listdir(f'{root}/sequences')
        for seq_name in seq_names:
            example_container = examples_train if seq_name in split['train'] else examples_test
            seq_dir = f'{root}/sequences/{seq_name}'
            info = spu.load_json(f'{seq_dir}/info.json')
            body_model = smpl_male if info['gender'] == 'male' else smpl_female

            i_cameras = info['kinects']
            cameras = [load_camera(
                intr_path=f'{seq_dir}/{info["intrinsic"]}/{i_cam}/calibration.json',
                extr_path=f'{seq_dir}/{info["config"]}/{i_cam}/config.json')
                for i_cam in i_cameras]

            snapshot_dirs = [x for x in glob.glob(f'{seq_dir}/*') if osp.isdir(x)]
            pose_sampler = AdaptivePoseSampler2(100, True, True, 100)

            for snapshot_dir in sorted(snapshot_dirs):
                world_coords_openpose = np.array(
                    spu.load_json(f'{snapshot_dir}/person/person_J3d.json')['body_joints3d'],
                    dtype=np.float32).reshape(25, 4)[:, :3] * 1000
                fit = spu.load_pickle(f'{snapshot_dir}/person/fit02/person_fit.pkl')
                world_coords_smpl = get_smpl_joints(body_model, fit)

                world_coords = np.concatenate([
                    world_coords_smpl[smpl_reordering], world_coords_openpose[openpose_reordering]],
                    axis=0)

                if pose_sampler.should_skip(world_coords):
                    continue

                for i_cam, camera in zip(i_cameras, cameras):
                    image_path = f'{snapshot_dir}/k{i_cam}.color.jpg'
                    image_relpath = osp.relpath(image_path, root)

                    mask_path = f'{snapshot_dir}/k{i_cam}.person_mask.jpg'
                    mask = imageio.imread(mask_path)

                    imcoords = camera.world_to_image(world_coords)
                    bbox = get_bbox(imcoords, image_relpath, all_detections)
                    if boxlib.area(bbox) == 0:
                        continue

                    ex = ds3d.Pose3DExample(
                        f'behave/{image_relpath}', world_coords, bbox=bbox, camera=camera,
                        mask=mask)
                    new_image_relpath = f'behave_downscaled/{image_relpath}'

                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        callback=example_container.append)

    examples_train.sort(key=lambda ex: ex.image_path)
    examples_test.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(
        joint_info, train_examples=examples_train, test_examples=examples_test)


def load_camera(intr_path, extr_path):
    extrinsics = spu.load_json(extr_path)
    R = np.array(extrinsics['rotation'], np.float32).reshape(3, 3).T
    t = np.array(extrinsics['translation'], np.float32) * 1000

    intrinsics = spu.load_json(intr_path)
    i = intrinsics['color']
    intrinsic_matrix = np.array([
        [i['fx'], 0, i['cx']],
        [0, i['fy'], i['cy']],
        [0, 0, 1]], np.float32)
    dist_coeffs = np.array(
        [i['k1'], i['k2'], i['p1'], i['p2'], i['k3'], i['k4'], i['k5'], i['k6']], np.float32)
    return cameravision.Camera(
        rot_world_to_cam=R, optical_center=t,
        intrinsic_matrix=intrinsic_matrix, distortion_coeffs=dist_coeffs, world_up=(0, -1, 0))


def make_joint_info():
    smpl_joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    smpl_reordering = [*range(1, 24), 0]
    smpl_joint_names = [smpl_joint_names[j] for j in smpl_reordering]
    smpl_edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,spin-bell-pelv-rhip-rkne-rank-rtoe'
    suf = '_coco'
    openpose_joint_names = (
        'nose,neck,rsho,relb,rwri,lsho,lelb,lwri,pelv,rhip,rkne,rank,lhip,'
        'lkne,lank,reye,leye,rear,lear,lfoo,ltoe,lhee,rfoo,rtoe,rhee'.split(','))
    openpose_reordering = [*range(8), *range(9, 25), 8]
    openpose_joint_names = [openpose_joint_names[j] + suf for j in openpose_reordering]
    openpose_edges = ('nose-neck-pelv-rhip-rkne-rank-rfoo-rtoe,rank-rhee,nose-reye-rear,'
                      'neck-rsho-relb-rwri')
    openpose_edges = openpose_edges.replace('-', suf + '-').replace(',', suf + ',') + suf
    return JointInfo(
        smpl_joint_names + openpose_joint_names,
        smpl_edges + ',' + openpose_edges), smpl_reordering, openpose_reordering


def get_bbox(im_coords, image_relpath, boxes):
    bbox = boxlib.expand(boxlib.bb_of_points(im_coords), 1.05)

    if image_relpath in boxes and boxes[image_relpath]:
        candidates = boxes[image_relpath]
        ious = np.array([boxlib.iou(b, bbox) for b in candidates])
        i_best = np.argmax(ious)
        if ious[i_best] > 0.5:
            bbox = candidates[i_best]
    else:
        logger.info(f'No detection {image_relpath}')

    return boxlib.intersection(bbox, boxlib.full(imsize=[2048, 1536]))


def get_smpl_joints(body_model, fit):
    return body_model.single(
        fit['pose'], fit['betas'], fit['trans'],
        return_vertices=False)['joints'] * 1000


if __name__ == '__main__':
    main()
