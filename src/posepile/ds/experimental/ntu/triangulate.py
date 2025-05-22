import os.path

import posepile.ds.experimental.ntu.main as ntu_main
import simplepyutils as spu
from posepile.ds.experimental.ntu.main import NTU_ROOT
from posepile.ds.experimental.triangulate_common import *
from posepile.paths import DATA_ROOT


def main():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    ignores = spu.read_lines(f'{NTU_ROOT}/ignore_videos.txt')
    rgb_paths = spu.sorted_recursive_glob(f'{NTU_ROOT}/nturgb+d_rgb/**/*_rgb.avi')
    video_ids = sorted(ntu_main.get_video_id(p) for p in rgb_paths)
    video_ids = [
        v for v in video_ids if not any(f'{v[:4]}C{i + 1:03d}{v[8:]}' in ignores for i in range(3))]

    video_id_to_calib_id = spu.load_pickle(
        f'{NTU_ROOT}/camcalib/video_id_to_calib_id_manualfix.pkl')

    def get_calib3_id(video_id):
        video_ids_of_triplet = [f'{video_id[:4]}C{i + 1:03d}{video_id[8:]}' for i in range(3)]
        calib_ids = [video_id_to_calib_id[v] if v in video_id_to_calib_id else (v[:8] + 'V001')
                     for v in video_ids_of_triplet]
        return ''.join(calib_ids)

    video_ids_per_calib = spu.groupby(video_ids, get_calib3_id)
    print(len(video_ids_per_calib))  # 51
    calib_id, video_ids_task = list(video_ids_per_calib.items())[i_task]
    print(calib_id)

    video_ids_per_seq = list(spu.groupby(video_ids_task, ntu_main.get_seq_id).values())
    assert all(len(vs) == 3 for vs in video_ids_per_seq)

    camdict = spu.load_pickle(f'{NTU_ROOT}/cameras.pkl')
    joint_info = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    joint_info_gt = ntu_main.get_kinect_joint_info()
    two_person_actions = [*range(50, 61), *range(106, 121)]

    print('Loading GT...')
    gt_per_video = {
        k: ntu_main.examples_to_tracks(v)
        for k, v in spu.groupby(ntu_main.make_stage1(), lambda x: x['video_id']).items()
        if k in video_ids}
    assert len(gt_per_video) == len(video_ids)

    print('Arranging per video...')
    poses3d_boxes_gt_per_seq = [[None, None, None] for _ in range(len(video_ids_per_seq))]
    with spu.ThrottledPool() as pool:
        for i_seq, video_ids_per_cam in enumerate(spu.progressbar(video_ids_per_seq)):
            for i_cam, video_id in enumerate(video_ids_per_cam):
                pool.apply_async(
                    arrange_data,
                    (video_id, camdict[video_id[:8]], gt_per_video[video_id],
                     two_person_actions, joint_info, joint_info_gt),
                    callback=spu.itemsetter(poses3d_boxes_gt_per_seq, i_seq, i_cam))

    print('Arranging per cam...')
    try:
        poses3d_per_seq = [
            [poses for poses, boxes, gt in poses3d_boxes_gt_per_cam]
            for poses3d_boxes_gt_per_cam in poses3d_boxes_gt_per_seq
            if poses3d_boxes_gt_per_cam[0][0].shape[1] == 1]
        resamp_per_seq = [resample3_by_len(poses3d_per_cam)[0]
                          for poses3d_per_cam in poses3d_per_seq]
        resamp_per_cam_allseq = [np.concatenate([r[i] for r in resamp_per_seq]) for i in range(3)]
    except:
        poses3d_per_seq = [
            [poses for poses, boxes, gt in poses3d_boxes_gt_per_cam]
            for poses3d_boxes_gt_per_cam in poses3d_boxes_gt_per_seq]
        resamp_per_seq = [resample3_by_len(poses3d_per_cam)[0]
                          for poses3d_per_cam in poses3d_per_seq]
        resamp_per_cam_allseq = [np.concatenate([r[i] for r in resamp_per_seq]) for i in range(3)]

    cameras = [camdict[video_ids_per_seq[0][i_cam][:8]] for i_cam in range(3)]
    print('Calibrating...')
    cameras_calib = [
        cameras[0],
        calibrate_from_poses(resamp_per_cam_allseq[0], resamp_per_cam_allseq[1], cameras[1]),
        calibrate_from_poses(resamp_per_cam_allseq[0], resamp_per_cam_allseq[2], cameras[2])]

    print('Triangulating...')
    with spu.ThrottledPool() as pool:
        for video_ids_per_cam, poses3d_boxes_gt_per_cam in zip(
                spu.progressbar(video_ids_per_seq), poses3d_boxes_gt_per_seq):
            pool.apply_async(
                process_sequence, (video_ids_per_cam, poses3d_boxes_gt_per_cam, cameras_calib))


def arrange_data(
        video_id, camera, gt, two_person_actions, joint_info, joint_info_gt, pred_dir='pred'):
    v = video_id
    preds = spu.load_pickle(f'{NTU_ROOT}/{pred_dir}/{v[:4]}/{v[4:12]}/{v}_rgb.pkl')
    masks = spu.load_pickle(f'{NTU_ROOT}/stcn_pred/{v[:4]}/{v[4:12]}/{v}_rgb.pkl')
    n_people = 2 if int(video_id[-3:]) in two_person_actions else 1

    return pred_to_masked_avg_poses_assoc(
        preds['poses3d'], gt, preds['boxes'], masks, camera, n_people, joint_info,
        joint_info_gt, 60)


def process_sequence(video_ids_per_cam, poses3d_boxes_gt_per_cam, cameras):
    poses3d_per_cam, boxes_per_cam, gt3d_per_cam = zip(*poses3d_boxes_gt_per_cam)
    out_paths = [f'{NTU_ROOT}/triang/{k[:4]}/{k[4:12]}/{k}_rgb.pkl' for k in video_ids_per_cam]
    for i_ref_cam, out_path in enumerate(out_paths):
        if False and spu.is_pickle_readable(out_path):
            continue

        triangulate_sequence(
            i_ref_cam, out_path, poses3d_per_cam, gt3d_per_cam, boxes_per_cam, cameras,
            dynamic_time_warping=False, already_calibrated=True)


def pad_to_twoperson(x):
    if x.shape[1] == 1:
        return np.concatenate([x, np.full_like(x, fill_value=np.nan)], axis=1)
    else:
        return x


if __name__ == '__main__':
    main()
