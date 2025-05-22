import argparse
import os

import cameralib
import imageio.v2 as imageio
import more_itertools
import numpy as np
import poseviz
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.ds.experimental.cwi.triangulate as cwi_triangulate
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--video-suffix', type=str)
    parser.add_argument('--write-video', action=spu.argparse.BoolAction)
    parser.add_argument('--camera', type=str, default='free')
    parser.add_argument('--bone-length-file', type=str)
    parser.add_argument('--high-quality-viz', action=spu.argparse.BoolAction)
    parser.add_argument('--scale-align', action=spu.argparse.BoolAction)
    parser.add_argument('--skeleton-types-file', type=str)
    parser.add_argument('--skeleton', type=str, default='smpl+head_30')
    spu.initialize(parser)

    ji3d = get_joint_info()
    index_selector = spu.load_pickle(FLAGS.skeleton_types_file)[FLAGS.skeleton]['indices']
    root_joint_name = 'pelv_smpl' if 'pelv_smpl' in ji3d.ids else 'pelv'
    i_pelvis = ji3d.ids[root_joint_name]

    root = f'{DATA_ROOT}/chico'
    n_views = 3
    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, FLAGS.camera, n_views=n_views,
        world_up=(0, 0, 1), show_field_of_view=False, resolution=(1920, 1080), queue_size=1,
        ground_plane_height=-500, high_quality=FLAGS.high_quality_viz)

    camera_names, cameras, undist_cameras = load_cameras(root)

    seq_names = os.listdir(f'{root}/dataset_raw')
    for seq_name in seq_names:
        if FLAGS.write_video:
            out_video_path = f'{FLAGS.out_video_path}/{seq_name}{FLAGS.video_suffix}.mp4'
            viz.new_sequence_output(out_video_path, fps=25)

        video_readers = load_videos(seq_name)
        preds = load_preds(seq_name)

        framess = chunked_roundrobin(video_readers)
        boxess = chunked_roundrobin([p['boxes'] for p in preds])
        posess = chunked_roundrobin([p['poses3d'] for p in preds])

        for frame_per_cam, boxes_per_cam, poses_per_cam in zip(framess, boxess, posess):
            camposes_per_cam = [np.array(poses)[:, :, index_selector] for poses in poses_per_cam]
            world_poses_per_cam = np.array([
                c.camera_to_world(p) for c, p in zip(cameras, camposes_per_cam)])
            world_pose_per_cam = cwi_triangulate.find_main_person(
                world_poses_per_cam, ji3d, root_name=root_joint_name, n_aug=5, distance_thresh=1500)
            campose_per_cam = np.array([
                c.world_to_camera(p) for c, p in zip(cameras, world_pose_per_cam)])
            is_valid = np.logical_not(np.any(np.isnan(campose_per_cam), axis=(1, 2, 3)))

            if np.any(is_valid):
                campose_per_cam_valid = campose_per_cam[is_valid]
                cameras_valid = [c for c, v in zip(cameras, is_valid) if v]
                campose_per_cam_valid = cwi_triangulate.mask_and_average(
                    campose_per_cam_valid, 30)
                triang = cwi_triangulate.triangulate_poses(
                    cameras_valid, campose_per_cam_valid, imshape=(1512, 2688), min_inlier_views=2)
            else:
                triang = np.full(shape=[ji3d.n_joints, 3], dtype=np.float32, fill_value=np.nan)

            # Convert to camera coords, adjust scale/distance to ground truth, convert back
            # to world
            pelv_true = triang[i_pelvis]

            if FLAGS.scale_align and not np.any(np.isnan(pelv_true)):
                true_pelvis_depths = [c.world_to_camera(pelv_true)[2] for c in cameras]
                camposes_per_cam = [p[np.newaxis] for p in campose_per_cam]
                camposes_per_cam_adjusted = [
                    poses * true_pelvis_depth / poses[..., i_pelvis:i_pelvis + 1, 2:]
                    for poses, true_pelvis_depth in zip(camposes_per_cam, true_pelvis_depths)]
                poses_adjusted = [
                    c.camera_to_world(poses)
                    for poses, c in zip(camposes_per_cam_adjusted, cameras)]
            else:
                poses_adjusted = world_poses_per_cam

            undist_frames = [
                cameralib.reproject_image(f, c, uc, output_imshape=[720, 1280])
                for f, c, uc in zip(frame_per_cam, cameras, undist_cameras)]
            undist_boxes = [
                np.array([cameralib.reproject_box(b, c, uc) for b in boxes], np.float32)
                for boxes, c, uc in zip(boxes_per_cam, cameras, undist_cameras)]
            view_infos = [
                poseviz.ViewInfo(
                    frame, boxes, np.mean(poses, axis=-3), c)  # , poses_true=triang[np.newaxis])
                for i, (frame, boxes, poses, c) in enumerate(
                    zip(undist_frames, undist_boxes, poses_adjusted, undist_cameras))]
            viz.update_multiview(view_infos)

    viz.close()


def load_videos(seq_name):
    video_paths = [
        f'{DATA_ROOT}/chico/dataset_raw/{seq_name}/{cam_name}.mp4'
        for cam_name in ['00_03', '00_06', '00_12']]
    video_readers = [
        imageio.get_reader(p, 'ffmpeg', output_params=['-map', '0:v:0'])
        for p in video_paths]
    return video_readers


def load_preds(seq_name):
    pred_paths = [
        f'{DATA_ROOT}/chico/1c6f6193_pred/dataset_raw/{seq_name}/{cam_name}.pkl'
        for cam_name in ['00_03', '00_06', '00_12']]
    return [spu.load_pickle(p) for p in pred_paths]


def load_cameras(root):
    cameras_dict = spu.load_pickle(f'{root}/recalibrated_cameras.pkl')
    camera_names = ['00_03', '00_06', '00_12']
    cameras = [cameras_dict[name] for name in camera_names]
    undist_cameras = [c.copy() for c in cameras]
    for uc in undist_cameras:
        uc.undistort()
        uc.scale_output(720 / 1512)
    return camera_names, cameras, undist_cameras


def chunked_roundrobin(iters):
    return more_itertools.chunked(more_itertools.roundrobin(*iters), len(iters))


def get_joint_info(skeleton=None):
    if skeleton is None:
        skeleton = FLAGS.skeleton

    d = spu.load_pickle(FLAGS.skeleton_types_file)[skeleton]
    joint_names = d['names']
    edges = d['edges']
    return JointInfo(joint_names, edges)


if __name__ == '__main__':
    main()
