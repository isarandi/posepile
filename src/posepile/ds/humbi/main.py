import argparse
import glob
import os.path as osp
import re

import boxlib
import cameravision
import more_itertools
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS
from smpl.numpy import SMPL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)
    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache('humbi_stage1.pkl', min_time="2020-09-13T20:00:00")
def make_stage1():
    root = f'{DATA_ROOT}/humbi'
    joint_info, smpl_reordering, openpose_reordering = make_joint_info()
    all_detections = spu.load_pickle(f'{root}/yolov4_detections.pkl')
    ignore_paths = spu.read_lines(f'{root}/ignore_images.txt')

    all_detections = {k: [np.array(x[:4]) for x in v] for k, v in all_detections.items()}
    examples = []
    body_model = SMPL(model_root=f'{DATA_ROOT}/body_models/smpl')

    with spu.ThrottledPool() as pool:
        snapshot_dirs = [x for x in glob.glob(f'{root}/subject_*/body/*') if osp.isdir(x)]
        for snapshot_dir in spu.progressbar(snapshot_dirs):
            cameras = load_cameras(f'{snapshot_dir}/..')
            smpl_params = np.loadtxt(f'{snapshot_dir}/reconstruction/smpl_parameter.txt')
            smpl_joints = get_smpl_joints(body_model, smpl_params)
            openpose_coords = np.genfromtxt(f'{snapshot_dir}/reconstruction/keypoints.txt')
            world_coords = np.concatenate([
                smpl_joints[smpl_reordering], openpose_coords[openpose_reordering]], axis=0) * 1000

            for image_path in glob.glob(f'{snapshot_dir}/image/image*.jpg'):
                if osp.relpath(image_path, root) in ignore_paths:
                    continue
                image_relpath = osp.relpath(image_path, root)
                i_cam = int(re.search(r'image(?P<num>\d+)\.jpg$', image_relpath)['num'])
                try:
                    camera = cameras[i_cam]
                except KeyError:
                    print(f'Cannot load camera {i_cam}, {snapshot_dir}.'
                          f' Such errors are expected. Continuing normally.')
                    continue
                imcoords = camera.world_to_image(world_coords)
                bbox = get_bbox(imcoords, image_relpath, all_detections)
                if boxlib.area(bbox) == 0:
                    continue
                ex = ds3d.Pose3DExample(
                    f'humbi/{image_relpath}', world_coords, bbox=bbox, camera=camera)
                new_image_relpath = f'humbi_downscaled/{image_relpath}'
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath), dict(ignore_broken_image=True),
                    callback=examples.append)

    # Here we filter out the ones which had broken images
    examples = [ex for ex in examples if ex is not None]
    examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, examples)


@spu.picklecache('humbi.pkl', min_time="2021-12-04T20:56:48")
def make_dataset():
    ds = make_stage1()
    mask_paths = glob.glob(f'{DATA_ROOT}/humbi_downscaled/masks/*.pkl')
    mask_dict = {}
    for path in mask_paths:
        mask_dict.update(spu.load_pickle(path))

    print(len(ds.examples[0]))
    for ex in ds.examples[0]:
        relpath = spu.last_path_components(ex.image_path, 5)
        ex.mask = mask_dict[relpath]
    return ds


def make_joint_info():
    smpl_joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    smpl_reordering = [*range(1, 24), 0]
    smpl_joint_names = [smpl_joint_names[j] for j in smpl_reordering]
    smpl_edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe'
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
            bbox = boxlib.box_hull(candidates[i_best], bbox)
    else:
        print(f'No detection {image_relpath}')

    return boxlib.intersection(bbox, boxlib.full(imsize=[1920, 1080]))


def get_smpl_joints(body_model, smpl_params):
    scale, trans, pose_params, shape_params = np.split(smpl_params, [1, 4, 76])
    result = body_model(pose_params[np.newaxis], shape_params[np.newaxis], return_vertices=False)
    joints = result['joints'][0] * scale[0] + trans
    return joints


def load_cameras(dirpath):
    extrinsic_lines = spu.read_lines(f'{dirpath}/extrinsic.txt')
    intrinsic_lines = spu.read_lines(f'{dirpath}/intrinsic.txt')
    cameras = {}

    for extr in more_itertools.chunked(extrinsic_lines[3:], 5):
        i_camera = int(extr[0].split()[1])
        optical_center = np.array([float(x) for x in extr[1].split()])
        R = np.array([[float(x) for x in l.split()] for l in extr[2:5]])
        cameras[i_camera] = cameravision.Camera(
            rot_world_to_cam=R, optical_center=optical_center * 1000, world_up=(0, -1, 0))

    for intr in more_itertools.chunked(intrinsic_lines[3:], 4):
        i_camera = int(intr[0].split()[1])
        cameras[i_camera].intrinsic_matrix = np.array(
            [[float(x) for x in l.split()] for l in intr[1:4]])
    return cameras


if __name__ == '__main__':
    main()
