import numpy as np
import posepile.ds.experimental.ntu.main as ntu_main
import posepile.ds.experimental.ntu.triangulate as ntu_triangulate
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.ds.bml_movi.main import solve_for_affine_weights
from posepile.paths import DATA_ROOT
from posepile.util import geom3d
from posepile.ds.experimental.ntu.main import NTU_ROOT


def main():
    gt2d_v, pred2d_v = load_matching_gt2d_and_latent_pred2d()
    pose_sampler = AdaptivePoseSampler(50, True, True)
    indices = [i for i, pose in enumerate(spu.progressbar(pred2d_v)) if
               not pose_sampler.should_skip(pose)]
    print(f'Got {len(indices)} input-output pose correspondences.')
    pred2d_v = pred2d_v[indices][::50]
    gt2d_v = gt2d_v[indices][::50]
    W = solve_for_affine_weights(pred2d_v, gt2d_v)

    # Filter out the outliers and re-estimate
    mapped_pred2d = geom3d.convert_pose(pred2d_v, W)
    good = np.all(np.linalg.norm(mapped_pred2d - gt2d_v, axis=-1) < 50, axis=-1)
    W = solve_for_affine_weights(pred2d_v[good], gt2d_v[good])

    np.save(f'{NTU_ROOT}/latent_to_kinect.npy', W)
    return W


def load_matching_gt2d_and_latent_pred2d():
    ignores = spu.read_lines(f'{NTU_ROOT}/ignore_videos.txt')
    rgb_paths = spu.sorted_recursive_glob(f'{NTU_ROOT}/nturgb+d_rgb/**/*_rgb.avi')
    video_ids = sorted(ntu_main.get_video_id(p) for p in rgb_paths)
    video_ids = [
        v for v in video_ids if
        not any(f'{v[:4]}C{i + 1:03d}{v[8:]}' in ignores for i in range(3))]

    camdict = spu.load_pickle(f'{NTU_ROOT}/cameras.pkl')
    joint_info = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    joint_info_gt = ntu_main.get_kinect_joint_info()

    print('Loading GT...')
    gt_per_video = {
        k: ntu_main.examples_to_tracks(v)
        for k, v in spu.groupby(ntu_main.make_stage1(), lambda x: x['video_id']).items()
        if k in video_ids}
    assert len(gt_per_video) == len(video_ids)
    two_person_actions = [*range(50, 61), *range(106, 121)]

    print('Arranging per video...')
    poses3d_boxes_gt_per_video = [None for _ in range(len(video_ids))]
    with spu.ThrottledPool() as pool:
        for i_vid, video_id in enumerate(spu.progressbar(video_ids)):
            pool.apply_async(
                ntu_triangulate.arrange_data,
                (video_id, camdict[video_id[:8]], gt_per_video[video_id],
                 two_person_actions, joint_info, joint_info_gt),
                callback=spu.itemsetter(poses3d_boxes_gt_per_video, i_vid))

    camera_per_video = [camdict[video_id[:8]] for video_id in video_ids]
    gts2d = np.array([
        camera.camera_to_image(gt) / np.nanmax(box[2:4]) * 256
        for (preds, boxes, gts), camera in
        zip(spu.progressbar(poses3d_boxes_gt_per_video), camera_per_video)
        for gts_in_frame, boxes_in_frame in zip(gts, boxes) if len(gts_in_frame) == 1
        for gt, box in zip(gts_in_frame, boxes_in_frame)])

    preds2d = np.array([
        camera.camera_to_image(pred) / np.nanmax(box[2:4]) * 256
        for (preds, boxes, gts), camera in
        zip(spu.progressbar(poses3d_boxes_gt_per_video), camera_per_video)
        for preds_in_frame, boxes_in_frame in zip(preds, boxes) if len(preds_in_frame) == 1
        for pred, box in zip(preds_in_frame, boxes_in_frame)])

    mask = np.logical_and(
        np.all(geom3d.are_joints_valid(preds2d), axis=-1),
        np.all(geom3d.are_joints_valid(gts2d), axis=-1))

    gt2d_v = np.array(gts2d)[mask]
    pred2d_v = np.array(preds2d)[mask]

    all_to_latent = np.load(f'{DATA_ROOT}/skeleton_conversion/all_to_latent_32_singlestage.npy')
    pred2d_latent = geom3d.convert_pose(pred2d_v, all_to_latent)

    return gt2d_v, pred2d_latent


def project(x):
    return x[..., :2] / x[..., 2:3]


if __name__ == '__main__':
    main()
