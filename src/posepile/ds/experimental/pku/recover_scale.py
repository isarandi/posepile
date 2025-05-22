import os.path as osp

import numpy as np
import posepile.ds.experimental.ntu.recover_scale as ntu_recover_scale
import posepile.util.rigid_alignment as rigid_alignment
import simplepyutils as spu
from posepile.ds.experimental.pku.triangulate import get_video_id
from posepile.ds.experimental.pku.main import PKU_ROOT
from posepile.paths import DATA_ROOT
from posepile.util import geom3d


def main():
    def has_all_cameras(video_id):
        return all(
            osp.exists(f'{PKU_ROOT}/RGB_VIDEO/{video_id[:4]}-{camname}.avi')
            for camname in 'MLR')

    rgb_paths = spu.sorted_recursive_glob(f'{PKU_ROOT}/RGB_VIDEO/*.avi')
    video_ids = [k for k in map(get_video_id, rgb_paths) if has_all_cameras(k)]

    all_to_latent = np.load(
        f'{DATA_ROOT}/skeleton_conversion/all_to_latent_32_singlestage.npy')
    latent_to_kinect = np.load(f'{DATA_ROOT}/ntu/latent_to_kinect.npy')
    all_to_kinect = all_to_latent @ latent_to_kinect
    ji_main = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    ref_bone_len = np.array(
        spu.load_pickle(f'{DATA_ROOT}/ntu/predictor_bone_length_prior.pkl'), np.float32)

    with spu.ThrottledPool() as pool:
        for video_id in spu.progressbar(video_ids):
            pool.apply_async(process, (video_id, all_to_kinect, ji_main, ref_bone_len))


def process(video_id, all_to_kinect, ji_main, ref_bone_len):
    out_path = f'{PKU_ROOT}/triang_scaled/{video_id}.pkl'
    if osp.exists(out_path):
        return
    triang = spu.load_pickle(f'{PKU_ROOT}/pred/{video_id}_triang.pkl')
    gts = triang['original_kinect_poses3d'].reshape([-1, 25, 3]) * 1000
    preds = triang['camera'].world_to_camera(
        geom3d.convert_pose(triang['poses3d'], all_to_kinect).reshape([-1, 25, 3]))
    preds_world = triang['poses3d'].reshape([-1, 122, 3])

    bones_valid = geom3d.are_bones_plausible(
        preds_world, ref_bone_len, ji_main,
        relsmall_thresh=0.3, relbig_thresh=1.7, absbig_thresh=150)

    mask = np.logical_and(
        np.all(
            np.logical_and(
                geom3d.are_joints_valid(gts),
                geom3d.are_joints_valid(preds)), axis=-1),
        np.all(bones_valid, axis=-1))

    print(np.mean(mask), mask.shape)

    gts = gts[mask]
    preds = preds[mask]

    scales = np.array([
        geom3d.get_scale(gt) / geom3d.get_scale(pred)
        for gt, pred in zip(gts, preds)])
    aligned = rigid_alignment.rigid_align_many(gts, preds, scale_align=True)
    good = np.all(np.linalg.norm(aligned - preds, axis=-1) < 100, axis=-1)
    scales = scales[good]
    scales = scales[np.logical_and(scales < 1.3, scales > 0.8)]
    scale_factor = np.median(scales)
    print(f'{video_id}: {len(scales)}, {scale_factor:.1%} factor')

    if np.isnan(scale_factor) or len(scales) < 50:
        scale_factor = 1
    else:
        scale_factor = np.clip(scale_factor, 0.8, 1.2)

    # Determine ground plane and from that the up vector
    ankles = preds_world[mask][:, [ji_main.ids.ltoe, ji_main.ids.rtoe]].reshape([-1, 3])
    ankles = ankles[geom3d.are_joints_valid(ankles)] * scale_factor
    if len(ankles) > 0:
        world_up = ntu_recover_scale.find_up_vector(ankles)
    else:
        world_up = np.array([0, -1, 0], np.float32)

    angle = np.rad2deg(np.arccos(np.dot([0, -1, 0], world_up)))
    print(f'Angle: {angle:.1f} deg')

    print(f'{video_id}: {world_up} up')

    triang['camera'].t *= scale_factor
    triang['camera'].world_up = world_up
    triang['poses3d'] *= scale_factor
    spu.dump_pickle(triang, out_path)


if __name__ == '__main__':
    main()
