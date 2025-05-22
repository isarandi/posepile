import argparse
import functools
import itertools
import os
import os.path as osp

import cameralib
import more_itertools
import numpy as np
import poseviz
import simplepyutils as spu
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_inputs as tfinp
from simplepyutils import FLAGS

import posepile.ds.experimental.cwi.triangulate as cwi_triangulate
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--write-video', action=spu.argparse.BoolAction)
    parser.add_argument('--camera', type=str, default='free')
    parser.add_argument('--bone-length-file', type=str)
    parser.add_argument('--high-quality-viz', action=spu.argparse.BoolAction)
    parser.add_argument('--scale-align', action=spu.argparse.BoolAction)
    parser.add_argument('--skeleton-types-file', type=str)
    parser.add_argument('--skeleton', type=str, default='smpl+head_30')
    spu.initialize(parser)

    ji3d = get_joint_info()
    model = tfhub.load(FLAGS.model_path)
    index_selector = spu.load_pickle(FLAGS.skeleton_types_file)[FLAGS.skeleton]['indices']
    n_views = 3
    viz = poseviz.PoseViz(
        ji3d.names, ji3d.stick_figure_edges, FLAGS.camera, n_views=n_views,
        world_up=(0, 0, 1), show_field_of_view=False, resolution=(1920, 1080), queue_size=1,
        ground_plane_height=0, high_quality=FLAGS.high_quality_viz) if FLAGS.viz else None

    cameras_dict = spu.load_pickle(f'{DATA_ROOT}/chico/recalibrated_cameras.pkl')
    camera_names = ['00_03', '00_06', '00_12']
    cameras = [cameras_dict[name] for name in camera_names]
    undist_cameras = [c.copy() for c in cameras]
    for uc in undist_cameras:
        uc.undistort()
        uc.scale_output(480 / 1512)

    intrinsics = [c.intrinsic_matrix for c in cameras]
    extrinsics = [c.get_extrinsic_matrix() for c in cameras]
    distortion_coeffs = [c.distortion_coeffs for c in cameras]

    predict_fn = functools.partial(
        model.detect_poses_batched, detector_threshold=0.35, detector_nms_iou_threshold=0.6,
        suppress_implausible_poses=False, internal_batch_size=110, world_up_vector=(0, 0, 1),
        detector_flip_aug=FLAGS.detector_flip_aug, num_aug=FLAGS.num_aug, average_aug=False,
        skeleton='', max_detections=1)

    seq_names = os.listdir(f'{DATA_ROOT}/chico/dataset_raw')
    for seq_name in seq_names:
        if FLAGS.viz and FLAGS.write_video:
            out_video_path = f'{FLAGS.out_video_path}/{seq_name}{FLAGS.video_suffix}.mp4'
            viz.new_sequence_output(out_video_path, fps=25)

        video_paths = [
            f'{DATA_ROOT}/chico/dataset_raw/{seq_name}/{cam_name}.mp4'
            for cam_name in camera_names]
        calib_ds = tf.data.Dataset.from_tensor_slices(
            (intrinsics, extrinsics, distortion_coeffs)).repeat()

        ds, frames_cpu = tfinp.interleaved_video_files(
            video_paths, extra_data=calib_ds, batch_size=32, tee_cpu=True)
        pred_stream = prediction_stream_gen(predict_fn, ds)
        chunked_pred_stream = more_itertools.chunked(pred_stream, n_views)
        chunked_frame_stream_cpu = more_itertools.chunked(
            itertools.chain.from_iterable(frames_cpu), n_views)

        triangs = []
        root_joint_name = 'pelv_smpl' if 'pelv_smpl' in ji3d.ids else 'pelv'

        for preds_per_cam, frame_per_cam in spu.progressbar(
                zip(chunked_pred_stream, chunked_frame_stream_cpu)):
            preds_per_cam = list(preds_per_cam)
            boxess, posess = zip(*preds_per_cam)
            world_poses_per_cam = [poses[:, :, index_selector] for poses in posess]
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

            triangs.append(triang)

            if FLAGS.viz:
                # Convert to camera coords, adjust scale/distance to ground truth, convert back
                # to world
                i_pelvis = ji3d.ids[root_joint_name]
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
                    cameralib.reproject_image(f, c, uc, output_imshape=[480, 854])
                    for f, c, uc in zip(frame_per_cam, cameras, undist_cameras)]
                undist_boxes = [
                    np.array([cameralib.reproject_box(b, c, uc) for b in boxes], np.float32)
                    for boxes, c, uc in zip(boxess, cameras, undist_cameras)]
                view_infos = [
                    poseviz.ViewInfo(
                        frame, boxes, np.mean(poses, axis=-3), c, poses_true=triang[np.newaxis])
                    for i, (frame, boxes, poses, c) in enumerate(
                        zip(undist_frames, undist_boxes, poses_adjusted, undist_cameras))]
                viz.update_multiview(view_infos)

    if viz is not None:
        viz.close()


def prediction_stream_gen(predict_fn, ds_batched):
    for frames, (intr, extr, dist) in ds_batched:
        pred = predict_fn(
            frames, intrinsic_matrix=intr, extrinsic_matrix=extr, distortion_coeffs=dist)

        yield from zip(pred['boxes'].numpy(), pred['poses3d'].numpy())


def get_joint_info(skeleton=None):
    if skeleton is None:
        skeleton = FLAGS.skeleton

    d = spu.load_pickle(FLAGS.skeleton_types_file)[skeleton]
    joint_names = d['names']
    edges = d['edges']
    return JointInfo(joint_names, edges)


if __name__ == '__main__':
    main()
