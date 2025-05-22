import argparse
import glob
import json
import os
import os.path
import os.path as osp
import re

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import posepile.util.videoproc as videoproc
import scipy.optimize
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    parser.add_argument('--detector-path', type=str)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_stage2()
    elif FLAGS.stage == 3:
        make_dataset()


# STAGE 1: filter which poses to use and generate task files for detection and segmentation
@spu.picklecache('panoptic_stage1_extra.pkl', min_time="2021-12-29T17:38:33")
def make_stage1():
    names = (
        'neck,nose,lsho,lelb,lwri,lhip,lkne,lank,rsho,relb,rwri,rhip,rkne,rank,leye,'
        'lear,reye,rear,pelv')
    edges = 'nose-neck-pelv,rwri-relb-rsho-neck,rank-rkne-rhip-pelv,nose-reye-rear'
    selection = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2]
    joint_info = JointInfo(names, edges)
    sequences = (
        '160224_haggling1 160226_haggling1 160422_haggling1 160906_band1 160906_band2 '
        '160422_ultimatum1 160906_band3 160906_band4 160906_pizza1 161029_build1 161029_car1 '
        '161029_flute1 161029_piano1 161029_piano2 161029_piano3 161029_piano4 '
        '161029_sports1 161029_tools1 161202_haggling1 170221_haggling_b1 '
        '170221_haggling_b2 170221_haggling_b3 170221_haggling_m1 170221_haggling_m2 '
        '170221_haggling_m3 170224_haggling_a1 170224_haggling_a2 170224_haggling_a3 '
        '170224_haggling_b1 170224_haggling_b2 170224_haggling_b3 170228_haggling_a1 '
        '170228_haggling_a2 170228_haggling_a3 170228_haggling_b1 170228_haggling_b2 '
        '170228_haggling_b3 170404_haggling_a1 170404_haggling_a2 170404_haggling_a3 '
        '170404_haggling_b1 170404_haggling_b2 170404_haggling_b3 170407_haggling_a1 '
        '170407_haggling_a2 170407_haggling_a3 170407_haggling_b1 170407_haggling_b2 '
        '170407_haggling_b3 170407_office2 170915_office1 171026_cello3 171026_pose1 '
        '171026_pose2 171026_pose3 171204_pose1 171204_pose2 171204_pose3 '
        '171204_pose4 171204_pose5 171204_pose6 160906_ian1 160906_ian2 160906_ian3 160906_ian5 '
        '170915_toddler5').split()

    root_path = f'{DATA_ROOT}/panoptic'
    examples = []

    for sequence_name in sequences:
        print(f'Processing sequence {sequence_name}...')
        seq_dir = f'{root_path}/{sequence_name}'
        video_dir = f'{seq_dir}/hdVideos'

        video_paths = sorted([f'{video_dir}/{filename}' for filename in os.listdir(video_dir)])
        cam_names = [p.split('.')[0][-5:] for p in video_paths]
        cameras = get_cameras(f'{seq_dir}/calibration_{sequence_name}.json', cam_names)

        coords_files1 = glob.glob(f'{seq_dir}/hdPose3d_stage1_coco19/body3DScene_*.json')
        coords_files2 = glob.glob(f'{seq_dir}/hdPose3d_stage1_coco19/hd/body3DScene_*.json')
        coord_files = sorted([*coords_files1, *coords_files2])
        prev_poses = []
        for coords_path in spu.progressbar(coord_files):
            i_frame = int(re.match(r'.+/body3DScene_(\d{8})\.json', coords_path).group(1))
            try:
                bodies = spu.load_json(coords_path)['bodies']
            except json.JSONDecodeError:
                print(f'Could not load {coords_path}')
                continue

            current_poses = []
            for body in bodies:
                arr = np.array(body['joints19']).reshape(-1, 4)
                # Multiplying by 10 because CMU-Panoptic uses centimeters and we use millimeters.
                world_coords = arr[selection, :3] * 10
                confidences = arr[selection, 3]
                world_coords[confidences <= 0.1] = np.nan
                if np.count_nonzero(~np.isnan(world_coords[:, 0])) >= 6:
                    current_poses.append(world_coords)

            are_changes_sufficient = are_changes_sufficient_and_update(prev_poses, current_poses)

            for world_coords, suf in zip(current_poses, are_changes_sufficient):
                for cam_name, video_path in zip(cam_names, video_paths):
                    im_path = f'{osp.splitext(video_path)[0]}/{i_frame}'
                    camera = cameras[cam_name]
                    im_coords = camera.world_to_image(world_coords)
                    bbox = boxlib.expand(boxlib.bb_of_points(im_coords), 1.25)
                    ex = ds3d.Pose3DExample(im_path, world_coords, bbox, camera)

                    # Check if enough joints are within the image frame bounds
                    enough_visible = np.count_nonzero(
                        camera.is_visible(world_coords, [1920, 1080])) >= 6
                    if suf and enough_visible:
                        ex.bbox = np.array([*ex.bbox, 1])
                    else:
                        ex.bbox = np.array([*ex.bbox, 0])

                    examples.append(ex)
        print(f'{len(examples)} example candidates so far.')
    ds = ds3d.Pose3DDataset(joint_info, examples)

    def get_video_path(ex):
        return spu.path_range(ex.image_path, 0, -1)

    examples_grouped_by_video = spu.groupby(ds.examples[0], get_video_path)
    for i, (video_path, examples) in enumerate(examples_grouped_by_video.items()):
        spu.dump_pickle(
            (video_path, examples), f'{DATA_ROOT}/panoptic_downscaled/tasks/task_{i:06d}.pkl')

    return ds


def fix_path(p):
    parts = p.split('/')
    video_path = '/'.join(parts[:-1])
    iframe = parts[-1]
    return video_path[2:-10] + '.mp4/' + iframe


# Stage2: generate the actual example images
def make_stage2():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    output_path = f'{DATA_ROOT}/panoptic_downscaled/tasks/processed_task_{task_id:06d}.pkl'
    if osp.exists(output_path):
        return

    task_paths = sorted(glob.glob(f'{DATA_ROOT}/panoptic_downscaled/tasks/task_*.pkl'))
    task_path = task_paths[task_id]
    video_id, examples = spu.load_pickle(task_path)
    video_path = video_id + '.mp4'

    # Missing or broken videos
    missing_hd20_videos = (
        '171204_pose3 171204_pose4 171204_pose5 160422_haggling1'.split())
    if any(f'{v}/hdVideos/hd_00_20.mp4' in video_path for v in missing_hd20_videos):
        return
    if '160906_band3/hdVideos/hd_00_03' in video_path:
        return

    detector = load_detector()
    print(video_path)
    print(len(examples))

    def get_i_frame(ex):
        return int(ex.image_path.split('/')[-1])

    frame_to_examples = spu.groupby(examples, get_i_frame)
    output_examples = []
    n_frames = videoproc.num_frames(video_path)

    with imageio.get_reader(video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(spu.progressbar(reader, total=n_frames)):
            gt_people = frame_to_examples[i_frame]

            # Skip frames that don't have any people in them based on the examples
            if all([ex.bbox[-1] == 0 for ex in gt_people]):
                continue
            detections = detector.predict_single_image(frame, 0.2, 0.7).numpy()
            iou_matrix = np.array([[boxlib.iou(gt_person.bbox[:4], box[:4])
                                    for box in detections]
                                   for gt_person in gt_people])
            gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

            for i_gt, i_det in zip(gt_indices, det_indices):
                gt_box = gt_people[i_gt].bbox
                det_box = detections[i_det]
                if (iou_matrix[i_gt, i_det] > 0.1 and
                        boxlib.area(det_box) < 2 * boxlib.area(gt_box) and
                        gt_box[-1] == 1):
                    ex = gt_people[i_gt]
                    ex.bbox = np.array(detections[i_det][:4])
                    image_prefix = ex.image_path.replace('.mp4', '')
                    new_image_path = f'{image_prefix}_{i_gt}.jpg'
                    new_image_path = new_image_path.replace(
                        f'{DATA_ROOT}/panoptic', f'{DATA_ROOT}/panoptic_downscaled')
                    new_image_relpath = osp.relpath(new_image_path, DATA_ROOT)
                    # temporary abuse of attribute, store the actual frame in the image path
                    ex.image_path = frame
                    output_examples.append(make_efficient_example(ex, new_image_relpath))
                    ex.image_path = None

    print(len(output_examples))
    spu.dump_pickle((video_path, output_examples), output_path)


# Stage3: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('panoptic.pkl', min_time="2021-12-30T01:12:20")
def make_dataset():
    ds = make_stage1()
    ds.examples[0].clear()
    import gc
    gc.collect()
    task_paths = sorted(
        glob.glob(f'{DATA_ROOT}/panoptic_downscaled/tasks/processed_task*.pkl'))

    for task_path in task_paths:
        video_path, examples = spu.load_pickle(task_path)
        ds.examples[0].extend(examples)

    ds3d.add_masks(ds, f'{DATA_ROOT}/panoptic_downscaled/masks/', 4)
    print(len(ds.examples[0]))
    return ds


# Panoptic without toddler and with only the mid-level cameras
@spu.picklecache('panoptic_limited.pkl',
                 min_time="2022-06-10T21:29:12")
def make_panoptic_limited():
    ds = make_dataset()
    mid_cameras = [3, 5, 9, 15, 18, 20, 22, 23, 24]

    def is_good_path(p):
        is_toddler = '_ian' in p or '_toddler' in p
        is_mid_camera = any(f'hd_00_{c:02d}' in p for c in mid_cameras)
        return not is_toddler and is_mid_camera

    print(len(ds.examples[0]))
    ds.examples[0] = [x for x in spu.progressbar(ds.examples[0]) if is_good_path(x.image_path)]
    print(len(ds.examples[0]))
    return ds


def safe_appender(lst):
    def fun(item):
        if item is not None:
            lst.append(item)

    return fun


def parse_camera(cam):
    R = np.array(cam['R'])
    t = np.array(cam['t']) * 10  # CMU Panoptic is in centimeters, we convert to millimeters
    eye = (-R.T @ t).reshape(-1)
    return cameralib.Camera(eye, R, cam['K'], cam['distCoef'], world_up=(0, -1, 0))


def get_cameras(json_path, cam_names):
    cameras = spu.load_json(json_path)['cameras']
    return {cam['name']: parse_camera(cam)
            for cam in cameras
            if cam['name'] in cam_names}


def sufficient_pose_change(prev_pose, current_pose):
    if prev_pose is None:
        return True
    dists = np.linalg.norm(prev_pose - current_pose, axis=-1)
    return np.count_nonzero(dists[~np.isnan(dists)] >= 100) >= 3


def are_changes_sufficient_and_update(prev_poses, current_poses):
    result = [True] * len(current_poses)
    if not prev_poses:
        prev_poses.extend(current_poses)
        return result

    def pose_distance(p1, p2):
        return np.nanmean(np.linalg.norm(p1 - p2, axis=-1))

    dist_matrix = np.array([[pose_distance(p1, p2)
                             for p1 in current_poses]
                            for p2 in prev_poses])
    prev_indices, current_indices = scipy.optimize.linear_sum_assignment(dist_matrix)

    for pi, ci in zip(prev_indices, current_indices):
        result[ci] = sufficient_pose_change(prev_poses[pi], current_poses[ci])
        if result[ci]:
            prev_poses[pi] = current_poses[ci]

    for i, current_pose in enumerate(current_poses):
        if i not in current_indices:
            prev_poses.append(current_pose)

    return result


def load_detector():
    import tensorflow_hub as tfhub
    return tfhub.load('https://github.com/isarandi/tensorflow-yolov4-tflite/releases'
                      '/download/v0.1.0/yolov4_416.tar.gz')


def get_kinect_sync_indices(seq_dir, seq_name, i_ref):
    sync_data = spu.load_json(f'{seq_dir}/ksynctables_{seq_name}.json')
    univ_times = [np.array(sync_data['kinect']['color'][f'KINECTNODE{i}']['univ_time'])
                  for i in range(1, 11)]
    t_starts = [next(x for x in u if x > 0) for u in univ_times]
    t_ends = [next(reversed(list(x for x in u if x > 0))) for u in univ_times]
    reference_times = [
        t for t in univ_times[i_ref]
        if t > 0 and all(start <= t <= end for start, end in zip(t_starts, t_ends))]
    indices = [[np.argmin(np.abs(u - ref_time)) for ref_time in reference_times]
               for u in univ_times]
    return indices


if __name__ == '__main__':
    main()
