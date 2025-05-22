import argparse
import glob
import os
import os.path as osp

import boxlib
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
import posepile.util.improc as improc
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.ds.experimental.cwi.main import temporal_median
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.util import geom3d

MPI08_ROOT = f'{DATA_ROOT}/mpi08'


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
    out_path = f'{DATA_ROOT}/mpi08_downscaled/tasks/task_result_{i_task:03d}.pkl'
    # if osp.exists(out_path):
    #    return

    joint_info_base = spu.load_pickle(f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    i_selected_joints = [
        i for i, name in enumerate(joint_info_base.names)
        if name.endswith(('_cmu_panoptic', '_coco')) or '_' not in name]
    joint_info = joint_info_base.select_joints(i_selected_joints)
    joint_info.update_names([x.replace('cmu_panoptic', 'coco') for x in joint_info.names])
    ref_bone_len = np.array(
        spu.load_pickle(f'{DATA_ROOT}/ntu/predictor_bone_length_prior.pkl'), np.float32)

    triang_paths_all = spu.sorted_recursive_glob(f'{MPI08_ROOT}/triang/**/*.pkl')
    triang_paths = triang_paths_all[i_task:(i_task + 1)]
    examples = []

    cameras_all = spu.load_pickle(f'{MPI08_ROOT}/cameras.pkl')

    with spu.ThrottledPool() as pool:
        for triang_path in spu.progressbar(triang_paths):
            triang_relpath = osp.relpath(triang_path, f'{MPI08_ROOT}/triang')
            video_dir = f'{MPI08_ROOT}/{osp.dirname(triang_relpath)}'
            video_paths = spu.sorted_recursive_glob(f'{video_dir}/*.avi')
            video_relpaths = [osp.relpath(p, MPI08_ROOT) for p in video_paths]
            cameras = [cameras_all[p] for p in video_relpaths]

            pred_paths = [spu.replace_extension(f'{MPI08_ROOT}/pred/{p}', '.pkl')
                          for p in video_relpaths]
            boxes_per_cam = [spu.load_pickle(p)['boxes'] for p in pred_paths]
            triangs = spu.load_pickle(triang_path)
            triangs = temporal_median(triangs)

            for video_relpath, boxes_per_frame, camera in zip(
                    video_relpaths, boxes_per_cam, cameras):
                pose_sampler = AdaptivePoseSampler2(100, True, True, 100)
                video_path = f'{MPI08_ROOT}/{video_relpath}'
                with imageio.get_reader(video_path, 'ffmpeg') as frames:
                    for i_frame, (frame, triang, boxes) in enumerate(
                            zip(frames, triangs, boxes_per_frame)):

                        if not np.all(geom3d.are_bones_plausible(
                                triang, ref_bone_len, joint_info_base,
                                relsmall_thresh=0.3, relbig_thresh=1.5, absbig_thresh=150)):
                            continue

                        box = get_box(triang, boxes, camera)
                        n_joints_valid = np.count_nonzero(camera.is_visible(triang, [1004, 1004]))
                        if (n_joints_valid < 122 // 4 or
                                box is None or
                                pose_sampler.should_skip(triang)):
                            continue

                        frame = enhance_green_room(frame)
                        new_image_replath = f'mpi08_downscaled/{video_relpath}/{i_frame:06d}.jpg'
                        ex = ds3d.Pose3DExample(
                            frame, triang[i_selected_joints], bbox=box, camera=camera)
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_replath),
                            callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    ds_partial = ds3d.Pose3DDataset(joint_info, examples)
    spu.dump_pickle(ds_partial, out_path)


@spu.picklecache('mpi08.pkl', min_time="2022-01-25T23:04:36")
def make_dataset():
    partial_paths = sorted(glob.glob(f'{DATA_ROOT}/mpi08_downscaled/tasks/task_result_*.pkl'))
    partial_dss = [spu.load_pickle(p) for p in partial_paths]
    main_ds = partial_dss[0]
    for ds in partial_dss[1:]:
        main_ds.examples[0].extend(ds.examples[0])
    ds3d.filter_dataset_by_plausibility(
        main_ds, relsmall_thresh=0.5, relbig_thresh=1.25, absbig_thresh=80)
    ds3d.add_masks(main_ds, f'{DATA_ROOT}/mpi08_downscaled/masks', 4)
    return main_ds


def enhance_green_room(frame):
    frame = improc.adjust_gamma(frame, 0.67, inplace=True)
    frame = improc.white_balance(frame, 110, 145)
    return frame


def get_box(pose, boxes, camera):
    imcoords = camera.world_to_image(pose)
    gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.02)
    if len(boxes) == 0:
        return None

    i_det = np.argmax([boxlib.iou(gt_box, det[:4]) for det in boxes])
    box = boxes[i_det][:4]
    if boxlib.iou(gt_box, box) < 0.1:
        return gt_box
    return boxlib.box_hull(box, gt_box)


if __name__ == '__main__':
    main()
