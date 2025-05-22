import glob
import os.path as osp

import posepile.ds.experimental.ntu.main as ntu_main
from posepile.ds.experimental.pku.main import PKU_ROOT
from posepile.ds.experimental.triangulate_common import *
from posepile.paths import DATA_ROOT


def main():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    def has_all_cameras(video_id):
        return all(
            osp.exists(f'{PKU_ROOT}/RGB_VIDEO/{video_id[:4]}-{camname}.avi')
            for camname in 'MLR')

    rgb_paths = sorted(glob.glob(f'{PKU_ROOT}/RGB_VIDEO/*.avi'))
    video_ids_all = [k for k in map(get_video_id, rgb_paths) if has_all_cameras(k)]
    video_ids_all = [k for k in video_ids_all]
    video_ids_per_seq_all = list(spu.groupby(video_ids_all, get_sequence_id).values())
    print(len(video_ids_per_seq_all))
    n_seq_per_task = 8
    video_ids_per_seq = video_ids_per_seq_all[i_task * n_seq_per_task:(i_task + 1) * n_seq_per_task]

    with spu.ThrottledPool() as pool:
        for video_ids_per_cam in spu.progressbar(video_ids_per_seq):
            L, M, R = video_ids_per_cam
            video_ids_per_cam = [M, L, R]
            pool.apply_async(process_sequence, (video_ids_per_cam,))


def process_sequence(video_ids_per_cam):
    pred_paths = [f'{PKU_ROOT}/pred/{k}.pkl' for k in video_ids_per_cam]
    out_paths_all = [spu.replace_extension(p, '_triang.pkl') for p in pred_paths]
    if all(spu.is_pickle_readable(p) for p in out_paths_all):
        print('All done')
        return

    camera = cameralib.Camera(
        intrinsic_matrix=[[1030, 0, 980], [0, 1030, 550], [0, 0, 1]], world_up=[0, -1, 0])
    preds = [spu.load_pickle(p) for p in pred_paths]
    gts = [get_gt_poses(k) for k in video_ids_per_cam]
    masks = [spu.load_pickle(f'{PKU_ROOT}/stcn_pred/{k}.pkl') for k in video_ids_per_cam]
    cameras = [camera, camera, camera]

    two_person_vids = [
        5, 6, 15, 16, 25, 26, 35, 36, 45, 46, 55, 56, 65, 66, 75, 76, 85, 86, 95, 96, 105, 106, 115,
        116, 125, 126, 135, 136, 145, 146, 155, 156, 165, 166, 175, 176, 185, 186, 195, 196, 205,
        206, 215, 216, 225, 226, 235, 236, 245, 246, 255, 256, 261, 262, 271, 272, 285, 286, 299,
        300, 309, 310, 319, 320, 329, 330, 339, 340, 349, 350, 359, 360]
    n_people = 2 if int(video_ids_per_cam[0][:4]) in two_person_vids else 1
    joint_info = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    joint_info_gt = ntu_main.get_kinect_joint_info()

    poses3d_boxes_gt_per_cam = [
        pred_to_masked_avg_poses_assoc(
            p['poses3d'], gt, p['boxes'], mask, camera, n_people, joint_info, joint_info_gt, 60)
        for p, gt, mask, camera, video_id in zip(preds, gts, masks, cameras, video_ids_per_cam)]
    poses3d_per_cam, boxes_per_cam, gt3d_per_cam = zip(*poses3d_boxes_gt_per_cam)

    for i_ref_cam, pred_path_ref in enumerate(pred_paths):
        out_path = spu.replace_extension(pred_path_ref, '_triang.pkl')
        if spu.is_pickle_readable(out_path):
            continue

        triangulate_sequence(
            i_ref_cam, out_path, poses3d_per_cam, gt3d_per_cam, boxes_per_cam, cameras,
            dynamic_time_warping=True)


def get_gt_poses(video_id):
    values = np.loadtxt(f'{PKU_ROOT}/PKU_Skeleton_Renew/{video_id}.txt')
    values = values.reshape([-1, 2, 25, 3])
    values[..., 1] *= -1
    values[values == 0] = np.nan
    return [
        np.array(
            [p for p in poses_of_frame
             if not np.all(np.isnan(p))], np.float32).reshape([-1, 25, 3])
        for poses_of_frame in values]


def get_sequence_id(video_id):
    return video_id[:4]


def get_video_id(path):
    return osp.basename(path)[:6]


if __name__ == '__main__':
    main()
