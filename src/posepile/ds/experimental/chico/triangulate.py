import os.path
import re

import simplepyutils as spu

import posepile.datasets3d as ds3d
from posepile.ds.experimental.chico.save_camconfig import load_cameras
from posepile.ds.experimental.cwi.triangulate import triangulate_poses
from posepile.ds.experimental.triangulate_common import *
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT

CHICO_ROOT = f'{DATA_ROOT}/chico'


def main():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    rgb_paths = spu.sorted_recursive_glob(f'{CHICO_ROOT}/dataset_raw/*/*.mp4')
    video_ids = [get_video_id(p) for p in rgb_paths]
    video_ids_per_seq = list(spu.groupby(video_ids, get_sequence_id).values())
    n_seq_per_task = 8
    video_ids_per_seq = video_ids_per_seq[i_task * n_seq_per_task:(i_task + 1) * n_seq_per_task]
    pred_dir = f'{DATA_ROOT}/1c6f6193_pred'

    with spu.ThrottledPool() as pool:
        for video_ids_per_cam in spu.progressbar(video_ids_per_seq):
            pool.apply_async(process_sequence, (pred_dir, video_ids_per_cam,))


def process_sequence(pred_dir, video_ids_per_cam):
    pred_paths = [f'{pred_dir}/{k}.pkl' for k in video_ids_per_cam]
    out_paths_all = [spu.replace_extension(p, '_triang.pkl') for p in pred_paths]
    if all(spu.is_pickle_readable(p) for p in out_paths_all):
        print('All done')
        return
    cameras = load_cameras()
    cameras = [cameras['00_03'], cameras['00_06'], cameras['00_12']]

    preds = [spu.load_pickle(p) for p in pred_paths]
    gt = get_gt_poses(video_ids_per_cam[0])
    joint_info = ds3d.get_joint_info('huge8')
    joint_info_gt = get_chico_joint_info()

    poses3d_boxes_gt_per_cam = [pred_dense(p['poses3d'], p['boxes'], 60) for p in preds]
    poses3d_per_cam, boxes_per_cam = zip(*poses3d_boxes_gt_per_cam)
    poses3d_per_cam = np.array(poses3d_per_cam, dtype=np.float32)

    for i_ref_cam, pred_path_ref in enumerate(pred_paths):
        out_path = spu.replace_extension(pred_path_ref, '_triang.pkl')
        if spu.is_pickle_readable(out_path):
            continue

        # TODO fix this, the code is incomplete
        triangulate_sequence(
            i_ref_cam, out_path, poses3d_per_cam, gt3d_per_cam, boxes_per_cam, cameras,
            dynamic_time_warping=False)


def triangulate_sequence(out_path, poses3d_per_cam, gt3d_per_cam, boxes_per_cam, cameras):
    cameras_calib = [
        cameras[0],
        calibrate_from_poses(poses3d_per_cam[0], poses3d_per_cam[1], cameras[1]),
        calibrate_from_poses(poses3d_per_cam[0], poses3d_per_cam[2], cameras[2])]

    triangulate_poses(cameras_calib, poses3d_per_cam, imshape=(1512, 2688))

    reproj_errors = np.stack([
        calc_reproj_error(triangs, resamps[i_cam], cameras_calib[i_cam])
        for i_cam in range(3)], axis=1)

    median_err_over_cameras = np.nanmedian(reproj_errors, axis=1)
    choices = infargmin(median_err_over_cameras, axis=0)
    triangs_combined = np.choose(choices[..., np.newaxis], triangs)

    triple_triang_is_good_in_all = np.all(reproj_errors[0] < 60, axis=0)
    triangs_combined[triple_triang_is_good_in_all] = triangs[0][
        triple_triang_is_good_in_all]

    min_err = infmin(median_err_over_cameras, axis=0)
    resamps_gt = [resample(gt3d, ind) for gt3d, ind in zip(gt3d_per_cam, indices)]
    content = dict(
        camera=cameras_calib[i_ref_cam],
        poses3d=triangs_combined,
        boxes=boxes_per_cam[i_ref_cam],
        original_kinect_poses3d=resamps_gt[i_ref_cam],
        errors=min_err,
        frame_indices=indices)

    spu.dump_pickle(content, out_path)


def pred_dense(pred3d, boxes_det, stdev_thresh=40):
    _, n_aug, n_joints, n_coord = pred3d[0].shape
    poses_out = np.full([len(pred3d), n_aug, n_joints, 3], fill_value=np.nan, dtype=np.float32)
    dets_out = np.full([len(pred3d), 5], fill_value=np.nan, dtype=np.float32)

    for i_frame, (poses_in_frame, dets_in_frame) in enumerate(zip(pred3d, boxes_det)):
        if len(poses_in_frame) > 0:
            poses_out[i_frame] = poses_in_frame[0]
            dets_out[i_frame] = dets_in_frame[0]

    return mask_and_average(poses_out, stdev_thresh), dets_out


def get_chico_joint_info():
    joint_names = 'pelv,rhip,rkne,rank,lhip,lkne,lank,nose,neck,rsho,relb,rwri,lsho,lelb,lwri'
    edges = 'pelv-rhip-rkne-rank,pelv-neck-nose,neck-rsho-relb-rwri'
    return JointInfo(joint_names, edges)


def get_gt_poses(video_id):
    m = re.match(r'S(?P<subj>\d\d)_(?P<action>.+?)/(?P<camera>\d\d_\d\d)', video_id)
    human_and_robot_poses = spu.load_pickle(
        f'{CHICO_ROOT}/chico_3d_skeleton/dataset/{m["subj"]}/{m["action"]}.pkl')
    human_poses = np.array([p[0] for p in human_and_robot_poses], dtype=np.float32)
    return human_poses


def get_video_id(path):
    return '/'.join(osp.split(path)[-2:])[:-4]


def get_sequence_id(video_id):
    return video_id.split('/')[0]


if __name__ == '__main__':
    main()
