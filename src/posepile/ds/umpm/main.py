import argparse
import glob
import os
import os.path as osp

import boxlib
import cameralib
import cv2
import ezc3d
import numpy as np
import posepile.datasets3d as ds3d
import scipy.optimize
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS

UMPM_ROOT = f'{DATA_ROOT}/umpm'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/umpm_downscaled/examples/examples_{i_task:06d}.pkl'
    if osp.exists(out_path):
        return

    seq_filepaths_all = sorted(glob.glob(f'{UMPM_ROOT}/*.json'))
    seq_filepath = seq_filepaths_all[i_task]
    all_detections = spu.load_pickle(f'{UMPM_ROOT}/yolov4_detections.pkl')
    examples = []

    with spu.ThrottledPool() as pool:
        seq_info = spu.load_json(seq_filepath)
        seq_name = seq_info['id']
        calib_name = seq_info['calib']
        for camname in 'fsrl':
            print(f'{seq_name}_{camname}')
            camera = load_camera(f'{UMPM_ROOT}/Calib/{calib_name}_{camname}.ini')
            # poses_ik = load_from_c3d(f'{UMPM_ROOT}/Groundtruth/{seq_name}_ik.c3d')
            poses_vm = load_from_c3d(f'{UMPM_ROOT}/Groundtruth/{seq_name}_vm.c3d')
            if poses_vm.shape[1] == 30:
                poses_per_person = np.split(poses_vm, 2, axis=1)
            elif poses_vm.shape[1] == 15:
                poses_per_person = [poses_vm]
            else:
                raise Exception(f'{seq_filepath}: {poses_vm.shape}')

            sampler_per_person = [
                AdaptivePoseSampler2(
                    100, check_validity=True, assume_nan_unchanged=True, buffer_size=300)
                for _ in poses_per_person]
            for i_frame, pose_per_person in enumerate(zip(*poses_per_person)):
                impath = f'{UMPM_ROOT}/Video/{seq_name}_{camname}/frame_{i_frame:06d}.jpg'
                imcoords_per_person = [camera.world_to_image(p) for p in pose_per_person]
                box_per_person = [
                    boxlib.expand(boxlib.bb_of_points(p), 1.15) for p in imcoords_per_person]

                dets = all_detections[osp.relpath(impath, UMPM_ROOT)]
                boxes = get_boxes(box_per_person, dets)

                for i_person, (box, world_coords, sampler) in enumerate(
                        zip(boxes, pose_per_person, sampler_per_person)):
                    relpose = world_coords - np.nanmean(world_coords, axis=0)
                    if box is None or sampler.should_skip(relpose):
                        continue

                    im_relpath_noext = osp.splitext(osp.relpath(impath, UMPM_ROOT))[0]
                    new_image_relpath = f'umpm_downscaled/{im_relpath_noext}_{i_person}.jpg'
                    ex = ds3d.Pose3DExample(impath, world_coords, box, camera)
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath), callback=examples.append)

    spu.dump_pickle(examples, out_path)


def get_boxes(gt_boxes, detections):
    if detections.size == 0:
        return [None for _ in gt_boxes]

    iou_matrix = np.array([[boxlib.iou(gt_box[:4], det[:4])
                            for det in detections]
                           for gt_box in gt_boxes])
    gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
    result_boxes = [None for _ in gt_boxes]
    for i_gt, i_det in zip(gt_indices, det_indices):
        if iou_matrix[i_gt, i_det] >= 0.5:
            result_boxes[i_gt] = detections[i_det][:4]
    return result_boxes


@spu.picklecache('umpm.pkl', min_time="2021-12-25T19:47:12")
def make_dataset():
    example_paths = glob.glob(f'{DATA_ROOT}/umpm_downscaled/examples/examples_*.pkl')
    examples = [ex for p in example_paths for ex in spu.load_pickle(p)]
    examples.sort(key=lambda ex: ex.image_path)

    names = 'pelv,neck,head,rsho,relb,rwri,lsho,lelb,lwri,rhip,lhip,rkne,lkne,rank,lank'
    edges = 'rank-rkne-rhip-pelv-neck-head,rwri-relb-rsho-neck'
    joint_info = JointInfo(names, edges)
    ds = ds3d.Pose3DDataset(joint_info, examples)
    ds3d.filter_dataset_by_plausibility(
        ds, relsmall_thresh=0.5, relbig_thresh=1.25, absbig_thresh=80,
        set_to_nan_instead_of_removal=True)
    ds3d.add_masks(
        ds, f'{DATA_ROOT}/umpm_downscaled/masks',
        relative_root=f'{DATA_ROOT}/umpm_downscaled')
    return ds


def load_from_c3d(path):
    c3d = ezc3d.c3d(path)
    coords = c3d['data']['points'].transpose(2, 1, 0)[..., :3].astype(np.float32)
    return coords[::2]


def load_camera(filepath):
    lines = spu.read_lines(filepath)
    values = [[float(x) for x in line.split(' ') if x] for line in lines]
    intrinsic_matrix = np.asarray(values[:3])
    distortion_coeffs = np.asarray(values[3])
    distortion_coeffs = [*distortion_coeffs, 0.0]
    rot_vector = np.asarray(values[4])
    t = np.asarray(values[5])
    R = cv2.Rodrigues(rot_vector)[0]
    return cameralib.Camera(
        intrinsic_matrix=intrinsic_matrix, rot_world_to_cam=R, trans_after_rot=t,
        distortion_coeffs=distortion_coeffs, world_up=(0, 0, 1))


if __name__ == '__main__':
    main()
