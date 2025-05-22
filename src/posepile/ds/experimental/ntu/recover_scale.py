import numpy as np
import posepile.ds.experimental.ntu.main as ntu_main
import posepile.util.rigid_alignment as rigid_alignment
import pyransac3d
import simplepyutils as spu
from posepile.ds.experimental.ntu.main import NTU_ROOT
from posepile.paths import DATA_ROOT
from posepile.util import geom3d


def main():
    ignores = spu.read_lines(f'{NTU_ROOT}/ignore_videos.txt')
    rgb_paths = spu.sorted_recursive_glob(f'{NTU_ROOT}/nturgb+d_rgb/**/*_rgb.avi')
    video_ids = sorted(ntu_main.get_video_id(p) for p in rgb_paths)
    video_ids = [
        v for v in video_ids if not any(f'{v[:4]}C{i + 1:03d}{v[8:]}' in ignores for i in range(3))]
    video_id_to_calib_id = spu.load_pickle(
        f'{NTU_ROOT}/camcalib/video_id_to_calib_id_manualfix.pkl')
    all_to_latent = np.load(f'{DATA_ROOT}/skeleton_conversion/all_to_latent_32_singlestage.npy')
    latent_to_kinect = np.load(f'{NTU_ROOT}/latent_to_kinect.npy')
    all_to_kinect = all_to_latent @ latent_to_kinect
    ji_main = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')

    def get_calib3_id(video_id):
        video_ids_of_triplet = [f'{video_id[:4]}C{i + 1:03d}{video_id[8:]}' for i in range(3)]
        calib_ids = [video_id_to_calib_id[v] if v in video_id_to_calib_id else (v[:8] + 'V001')
                     for v in video_ids_of_triplet]
        return ''.join(calib_ids)

    ref_bone_len = np.array(
        spu.load_pickle(f'{NTU_ROOT}/predictor_bone_length_prior.pkl'), np.float32)
    video_ids_per_calib = spu.groupby(video_ids, get_calib3_id)
    with spu.ThrottledPool() as pool:
        for calib_id, video_ids_task in video_ids_per_calib.items():
            pool.apply_async(
                process, (calib_id, video_ids_task, all_to_kinect, ji_main, ref_bone_len))


def process(calib_id, video_ids_task, all_to_kinect, ji_main, ref_bone_len):
    triangs = [
        spu.load_pickle(f'{NTU_ROOT}/triang/{v[:4]}/{v[4:12]}/{v}_rgb.pkl')
        for v in video_ids_task]
    gts = np.concatenate(
        [t['original_kinect_poses3d'].reshape([-1, 25, 3])
         for t in triangs], axis=0)
    preds = np.concatenate(
        [t['camera'].world_to_camera(
            geom3d.convert_pose(t['poses3d'], all_to_kinect).reshape([-1, 25, 3]))
            for t in triangs], axis=0)
    preds_world = np.concatenate(
        [t['poses3d'].reshape([-1, 122, 3])
         for t in triangs], axis=0)

    bones_valid = geom3d.are_bones_plausible(
        preds_world, ref_bone_len, ji_main,
        relsmall_thresh=0.3, relbig_thresh=1.7, absbig_thresh=150)

    mask = np.logical_and(
        np.all(
            np.logical_and(
                geom3d.are_joints_valid(gts),
                geom3d.are_joints_valid(preds)), axis=-1),
        np.all(bones_valid, axis=-1))

    print(np.mean(mask))

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
    print(f'{calib_id}: {len(scales)}, {scale_factor:.1%} factor')

    if np.isnan(scale_factor) or len(scales) < 50:
        scale_factor = 1
    else:
        scale_factor = np.clip(scale_factor, 0.8, 1.2)

    # Determine ground plane and from that the up vector
    ankles = preds_world[mask][:, [ji_main.ids.ltoe, ji_main.ids.rtoe]].reshape([-1, 3])
    ankles = ankles[geom3d.are_joints_valid(ankles)] * scale_factor
    if len(ankles) > 0:
        world_up = find_up_vector(ankles)
    else:
        world_up = np.array([0, -1, 0], np.float32)
    # print(f'{calib_id}: {world_up} up')

    for v in video_ids_task:
        triang = spu.load_pickle(f'{NTU_ROOT}/triang/{v[:4]}/{v[4:12]}/{v}_rgb.pkl')
        triang['camera'].t *= scale_factor
        triang['camera'].world_up = world_up
        triang['poses3d'] *= scale_factor
        spu.dump_pickle(triang, f'{NTU_ROOT}/triang_scaled/{v[:4]}/{v[4:12]}/{v}_rgb.pkl')


def find_up_vector(points, almost_up=(0, -1, 0), thresh_degrees=60):
    almost_up = np.asarray(almost_up, np.float32)

    plane1 = pyransac3d.Plane()
    _, best_inliers = plane1.fit(points, thresh=25, maxIteration=5000)
    if len(best_inliers) < 3:
        return

    world_up = np.asarray(fit_plane(points[best_inliers]), np.float32)
    if np.dot(world_up, almost_up) < 0:
        world_up *= -1

    if np.rad2deg(np.arccos(np.dot(world_up, almost_up))) > thresh_degrees:
        world_up = almost_up

    world_up = np.array(world_up, np.float32)
    return world_up


def fit_plane(points):
    points = np.asarray(points, np.float32)
    x = points - np.mean(points, axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return vt[-1]


if __name__ == '__main__':
    main()
