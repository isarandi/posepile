import glob
import os.path as osp

import boxlib
import cameravision
import numpy as np
import simplepyutils as spu

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
import smpl.numpy
import scipy.optimize


@spu.picklecache('tdpw.pkl', min_time="2021-07-09T12:26:16")
def make_dataset():
    root = f'{DATA_ROOT}/3dpw'
    body_joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    selected_joints = [*range(1, 24), 0]
    joint_names = [body_joint_names[j] for j in selected_joints]
    edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe'
    joint_info = JointInfo(joint_names, edges)
    detections_all = spu.load_pickle(f'{root}/yolov4_detections.pkl')

    def get_examples(phase, pool):
        result = []
        seq_filepaths = glob.glob(f'{root}/sequenceFiles/{phase}/*.pkl')
        for filepath in seq_filepaths:
            seq = spu.load_pickle(filepath)
            seq_name = seq['sequence']
            intrinsics = seq['cam_intrinsics']
            extrinsics_per_frame = seq['cam_poses']

            examples_of_seq = []
            for i_person, (
                    gender, poses_seq, betas, trans_seq, coords2d_seq, coords3d_seq,
                    camvalid_seq) in enumerate(zip(
                seq['genders'], seq['poses'], seq['betas'], seq['trans'], seq['poses2d'],
                seq['jointPositions'], seq['campose_valid'])):

                gender = dict(m='male', f='female')[gender]
                bm = smpl.numpy.get_cached_body_model('smpl', gender)
                for i_frame, (pose, trans, coords2d, coords3d, extrinsics,
                              campose_valid) in enumerate(
                    zip(poses_seq, trans_seq, coords2d_seq, coords3d_seq, extrinsics_per_frame,
                        camvalid_seq)):

                    if not campose_valid or np.all(coords2d == 0):
                        continue

                    impath = f'{root}/imageFiles/{seq_name}/image_{i_frame:05d}.jpg'
                    camera_raw = cameravision.Camera(
                        extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
                        world_up=(0, 1, 0))
                    imcoords = camera_raw.world_to_image(coords3d.reshape(-1, 3))

                    # move to camera frame because up vector is unreliable in world frame
                    pose, trans = bm.rototranslate(
                        extrinsics[:3, :3], extrinsics[:3, 3], pose, betas, trans)

                    params = dict(type='smpl', gender=gender, pose=pose, shape=betas, trans=trans)
                    camera = cameravision.Camera(intrinsic_matrix=intrinsics, world_up=(0, -1, 0))
                    bbox = boxlib.expand(boxlib.bb_of_points(imcoords), 1.15)
                    ex = ds3d.Pose3DExample(
                        impath, bbox=bbox, camera=camera, parameters=params, world_coords=None)
                    examples_of_seq.append(ex)

            grouped = spu.groupby(examples_of_seq, key=lambda ex: ex.image_path)
            for frame_path, gt_people in grouped.items():
                detections = detections_all[osp.relpath(frame_path, f'{root}/imageFiles')]
                iou_matrix = np.array(
                    [[boxlib.iou(det, ex.bbox) for det in detections] for ex in gt_people])
                gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

                for i_gt, i_det in zip(gt_indices, det_indices):
                    ex = gt_people[i_gt]
                    if iou_matrix[i_gt, i_det] > 0.1:
                        ex.bbox = np.array(detections[i_det][:4])
                        noext, ext = osp.splitext(osp.relpath(frame_path, root))
                        new_image_relpath = f'tdpw_downscaled/{noext}_{i_gt:03d}.jpg'
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_relpath),
                            kwargs=dict(min_time="2021-07-09T12:28:07"), callback=result.append)
        return result

    with spu.ThrottledPool() as pool:
        train_examples = get_examples('train', pool)
        val_examples = get_examples('validation', pool)
        test_examples = get_examples('test', pool)

    train_examples.sort(key=lambda ex: ex.image_path)
    val_examples.sort(key=lambda ex: ex.image_path)
    test_examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(
        joint_info, train_examples, val_examples, test_examples, compute_bone_lengths=False)


if __name__ == '__main__':
    make_dataset()
