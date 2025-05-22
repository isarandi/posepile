import argparse
import glob
import os.path as osp

import boxlib
import cameralib
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS

HUMAN4D_ROOT = f'{DATA_ROOT}/human4d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache('human4d_stage1.pkl', min_time="2022-01-19T00:29:35")
def make_stage1():
    image_paths = spu.natural_sorted(
        glob.glob(f'{HUMAN4D_ROOT}/**/Dump/color/*.png', recursive=True))

    def get_camera_name(path):
        return osp.basename(path).split('_')[1]

    images_by_camera_name = spu.groupby(image_paths, get_camera_name)

    examples = []
    cameras = {subj: load_cameras(subj) for subj in 'S1 S2 S3 S4'.split()}
    offsets = {subj: load_offsets(subj) for subj in 'S1 S2 S3 S4'.split()}
    detections_all = spu.load_pickle(f'{HUMAN4D_ROOT}/yolov4_detections.pkl')

    names = (
        'pelv,spin0,spin1,spin2,spin3,neck,neck2,head,htop,neck3,rsho,relb,rwri,rpinkie,rthu,'
        'neck4_del,lsho,lelb,lwri,lpinkie,lthu,rhip,rkne,rank,rfoo,rtoe,rtoe2,lhip,lkne,lank,'
        'lfoo,ltoe,ltoe2').split(',')
    i_selected_joints = [i for i, name in enumerate(names) if not name.endswith('_del')]
    names = [names[i] for i in i_selected_joints]
    edges = ('rthu-rpinkie-rwri-relb-rsho-neck3-spin3-neck-neck2-head-htop,'
             'spin3-spin2-spin1-spin0-pelv-rhip-rkne-rank-rfoo-rtoe-rtoe2')
    joint_info = JointInfo(names, edges)

    with spu.ThrottledPool() as pool:
        for camera_name, image_paths in images_by_camera_name.items():
            pose_sampler = AdaptivePoseSampler2(100, True, True, 100)

            for image_path in spu.progressbar(image_paths):
                parts = spu.split_path(image_path)
                timestamp = parts[-4]
                subj = parts[-5]
                delta = offsets[subj][timestamp] - 1
                camera = cameras[subj][camera_name]
                i_frame = int(parts[-1].split('_')[0])
                dumpdir = '/'.join(parts[:-2])

                if i_frame - delta < 0:
                    continue
                try:
                    world_coords = np.load(
                        f'{dumpdir}/gposes3d/{i_frame - delta}.npy').squeeze(0)[i_selected_joints]
                except FileNotFoundError:
                    print(f'Not found {dumpdir}/gposes3d/{i_frame - delta}.npy')
                    continue

                if pose_sampler.should_skip(world_coords):
                    continue

                imcoords = camera.world_to_image(world_coords)
                gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.2)
                detections = detections_all[osp.relpath(image_path, HUMAN4D_ROOT)]
                if detections.size > 0:
                    i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in detections])
                    box = detections[i_det][:4]
                else:
                    box = gt_box

                if boxlib.iou(gt_box, box) < 0.5:
                    continue

                if np.max(box[2:4]) < 30:
                    continue

                image_relpath = osp.relpath(image_path, DATA_ROOT)
                ex = ds3d.Pose3DExample(image_relpath, world_coords, bbox=box, camera=camera)
                new_image_replath = spu.replace_extension(
                    f'human4d_downscaled/{osp.relpath(image_path, HUMAN4D_ROOT)}', '.jpg')
                pool.apply_async(
                    make_efficient_example, (ex, new_image_replath), callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, train_examples=examples)


def load_cameras(subj):
    device_infos = spu.load_json(f'{HUMAN4D_ROOT}/{subj}/device_repository.json')
    result = {}
    for device_info in device_infos:
        camera_name = device_info['Device']
        intr = np.array(device_info['Color Intrinsics'][0]['1280x720'], np.float32).reshape([3, 3])
        extr_path = f'{HUMAN4D_ROOT}/{subj}/pose/{camera_name}.extrinsics'
        if not osp.exists(extr_path):
            continue
        mat = np.loadtxt(extr_path)
        result[camera_name] = cameralib.Camera(
            intrinsic_matrix=intr, rot_world_to_cam=mat[:3].T, optical_center=mat[3])
    return result


def load_offsets(subj):
    lines = spu.read_lines(f'{HUMAN4D_ROOT}/{subj}/offsets.txt')
    result = {}
    for line in lines:
        name, offset1, offset2 = line.split('\t')
        result[name] = int(offset1)
    return result


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('human4d.pkl', min_time="2021-12-04T20:56:48")
def make_dataset():
    return ds3d.add_masks(make_stage1(), f'{DATA_ROOT}/human4d_downscaled/masks', 5)


if __name__ == '__main__':
    main()
