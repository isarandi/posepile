import argparse
import glob
import os
import os.path as osp

import imageio.v2 as imageio
import numpy as np
import rlemasklib
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
import posepile.util.maskproc as maskproc
from posepile.paths import DATA_ROOT
from posepile.util import geom3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

PKU_ROOT = f'{DATA_ROOT}/pku'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    out_path = f'{DATA_ROOT}/pku_downscaled/tasks/task_result_{i_task:03d}.pkl'
    # if osp.exists(out_path):
    #    return

    joint_info_base = spu.load_pickle(
        f'{DATA_ROOT}/skeleton_conversion/joint_info_122.pkl')
    i_selected_joints = [
        i for i, name in enumerate(joint_info_base.names)
        if name.endswith(('_cmu_panoptic', '_coco')) or '_' not in name]
    joint_info = joint_info_base.select_joints(i_selected_joints)
    joint_info.update_names([x.replace('cmu_panoptic', 'coco') for x in joint_info.names])
    ref_bone_len = np.array(
        spu.load_pickle(f'{DATA_ROOT}/ntu/predictor_bone_length_prior.pkl'), np.float32)

    pred_paths_all = sorted(glob.glob(f'{PKU_ROOT}/triang_scaled/*.pkl'))
    pred_paths = pred_paths_all[i_task * 8:(i_task + 1) * 8]
    examples = []

    with spu.ThrottledPool() as pool:
        for pred_path in spu.progressbar(pred_paths):
            video_id = osp.splitext(osp.basename(pred_path))[0]
            video_masks = spu.load_pickle(f'{PKU_ROOT}/stcn_pred/{video_id}.pkl')
            data = spu.load_pickle(pred_path)
            camera = data['camera']

            video_world_coords = mask_bad(data['poses3d'], data['errors'], 35)
            video_boxes = data['boxes']
            n_people = data['poses3d'].shape[1]
            pose_samplers = [
                AdaptivePoseSampler2(100, True, True, 100) for _ in
                range(n_people)]

            with imageio.get_reader(f'{PKU_ROOT}/RGB_VIDEO/{video_id}.avi', 'ffmpeg') as frames:
                for i_frame, (frame, world_coords_per_person,
                              box_per_person, mask_per_person) in enumerate(
                    zip(frames, video_world_coords, video_boxes, video_masks)):

                    mask_union = rlemasklib.union(mask_per_person)
                    mask_union = maskproc.resize_mask(mask_union, frame.shape)
                    for i_person, (world_coords, box, pose_sampler) in enumerate(
                            zip(world_coords_per_person, box_per_person, pose_samplers)):

                        if not np.all(geom3d.are_bones_plausible(
                                world_coords, ref_bone_len, joint_info_base,
                                relsmall_thresh=0.3, relbig_thresh=1.5, absbig_thresh=150)):
                            continue

                        if pose_sampler.should_skip(world_coords):
                            continue

                        new_image_replath = (
                            f'pku_downscaled/{video_id}/{i_frame:06d}_{i_person}.jpg')

                        ex = ds3d.Pose3DExample(
                            frame, world_coords[i_selected_joints], bbox=box, camera=camera,
                            mask=mask_union)
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_replath),
                            callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    ds_partial = ds3d.Pose3DDataset(joint_info, examples)
    spu.dump_pickle(ds_partial, out_path)


@spu.picklecache('pku.pkl', min_time="2022-02-08T19:40:18")
def make_dataset():
    partial_paths = sorted(glob.glob(f'{DATA_ROOT}/pku_downscaled/tasks/task_result_*.pkl'))
    partial_dss = [spu.load_pickle(p) for p in partial_paths]
    main_ds = partial_dss[0]
    for ds in partial_dss[1:]:
        main_ds.examples[0].extend(ds.examples[0])
    ds3d.filter_dataset_by_plausibility(
        main_ds, relsmall_thresh=0.5, relbig_thresh=1.25, absbig_thresh=80)
    ds3d.add_masks(main_ds, f'{DATA_ROOT}/pku_downscaled/masks', 2)
    return main_ds


def mask_bad(triang, err, thresh=35):
    triang = triang.copy()
    triang[err > thresh] = np.nan
    return triang


if __name__ == '__main__':
    main()
