import argparse
import os
import os.path as osp

import boxlib
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.ds.experimental.cwi.main import temporal_median
from posepile.ds.experimental.vcliti.save_camconfig import load_camera
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.util import geom3d
from simplepyutils import FLAGS

VCLITI_ROOT = f'{DATA_ROOT}/vcliti'


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
    out_path = f'{DATA_ROOT}/vcliti_downscaled/tasks/task_result_{i_task:03d}.pkl'
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

    triang_paths_all = spu.sorted_recursive_glob(f'{VCLITI_ROOT}/triang/**/*.pkl')
    triang_paths = triang_paths_all[i_task:(i_task + 1)]

    preds_all = spu.load_pickle(f'{VCLITI_ROOT}/metrabs_pred.pkl')

    examples = []

    with spu.ThrottledPool() as pool:
        for triang_path in triang_paths:
            triang_relpath = osp.relpath(triang_path, f'{VCLITI_ROOT}/triang')
            data = spu.load_pickle(triang_path)
            triangs = temporal_median(data['triangs'])
            indices_per_cam = data['indices_per_cam']

            seq_relpath = osp.dirname(triang_relpath)
            seq_path = f'{VCLITI_ROOT}/{seq_relpath}'
            print(seq_path)
            cam_ids = [d[-1] for d in spu.sorted_recursive_glob(f'{seq_path}/D?')]
            cameras = [load_camera(seq_relpath, cam_id)[0] for cam_id in cam_ids]
            print(cam_ids)
            frame_paths_per_cam = [
                spu.sorted_recursive_glob(f'{seq_path}/D{cam_id}/*.jpg')
                for cam_id in cam_ids]
            boxes_per_cam = [
                [[b for b in preds_all[osp.relpath(path, VCLITI_ROOT)]['boxes'] if b[-1] > 0.5]
                 for path in frame_paths]
                for frame_paths in frame_paths_per_cam]

            print([len(x) for x in boxes_per_cam])
            print([max(x) for x in indices_per_cam])
            print([len(x) for x in frame_paths_per_cam])
            boxes_corresp_per_cam = [
                [boxes[i] for i in indices]
                for boxes, indices in zip(boxes_per_cam, indices_per_cam)]
            frame_paths_corresp_per_cam = [
                [frame_paths[i] for i in indices]
                for frame_paths, indices in zip(frame_paths_per_cam, indices_per_cam)]

            for frame_paths, boxes_per_frame, camera in zip(
                    frame_paths_corresp_per_cam, boxes_corresp_per_cam, cameras):
                pose_sampler = AdaptivePoseSampler2(100, True, True, 100)
                for i_frame, (frame_path, triang, boxes) in enumerate(
                        zip(frame_paths, spu.progressbar(triangs), boxes_per_frame)):

                    if not np.all(geom3d.are_bones_plausible(
                            triang, ref_bone_len, joint_info_base,
                            relsmall_thresh=0.3, relbig_thresh=1.5, absbig_thresh=150)):
                        continue

                    box = get_box(triang, boxes, camera)
                    n_joints_valid = np.count_nonzero(camera.is_visible(triang, [1920, 1080]))
                    if (n_joints_valid < 122 // 4 or
                            box is None or
                            pose_sampler.should_skip(triang)):
                        continue

                    relpath_to_ds = osp.relpath(frame_path, VCLITI_ROOT)
                    new_image_replath = f'vcliti_downscaled/{relpath_to_ds}'
                    ex = ds3d.Pose3DExample(
                        osp.relpath(frame_path, DATA_ROOT),
                        triang[i_selected_joints], bbox=box, camera=camera)
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_replath),
                        kwargs=dict(horizontal_flip=True),
                        callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    ds_partial = ds3d.Pose3DDataset(joint_info, examples)
    spu.dump_pickle(ds_partial, out_path)


@spu.picklecache('vcliti.pkl', min_time="2022-01-23T00:32:36")
def make_dataset():
    partial_paths = spu.sorted_recursive_glob(
        f'{DATA_ROOT}/vcliti_downscaled/tasks/task_result_*.pkl')
    partial_dss = [spu.load_pickle(p) for p in partial_paths]
    main_ds = partial_dss[0]
    for ds in partial_dss[1:]:
        main_ds.examples[0].extend(ds.examples[0])
    ds3d.filter_dataset_by_plausibility(
        main_ds, relsmall_thresh=0.5, relbig_thresh=1.25, absbig_thresh=80)
    ds3d.add_masks(
        main_ds, f'{DATA_ROOT}/vcliti_downscaled/masks',
        relative_root=f'{DATA_ROOT}/vcliti_downscaled')
    return main_ds


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
