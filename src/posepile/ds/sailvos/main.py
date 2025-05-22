import collections
import glob
import os.path as osp

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import rlemasklib
import scipy.optimize
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example

SAILVOS_ROOT = f'{DATA_ROOT}/sailvos'


@spu.picklecache('sailvos.pkl', min_time="2022-01-04T23:38:48")
def make_dataset():
    joint_info = JointInfo(
        'htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,spin,head,pelv,rtoe,'
        'ltoe,nose,reye,leye,rear,lear,rhan,lhan',
        'htop-head-neck-spin-pelv-rhip-rkne-rank-rtoe,neck-rsho-relb-rwri-rhan,pelv,nose-reye-rear')

    filepaths2d = spu.sorted_recursive_glob(f'{SAILVOS_ROOT}/**/*_2d.txt')
    image_to_examples = collections.defaultdict(list)

    def merge_examples(d):
        for k, v in d.items():
            image_to_examples[k] += v

    # Here we select and load the gt examples and eliminate redundant poses
    posefiles2d_per_seq = spu.groupby(filepaths2d, get_seq_name)
    with spu.ThrottledPool() as pool:
        for seq, seq_paths2d in spu.progressbar_items(posefiles2d_per_seq):
            print(seq, flush=True)
            pool.apply_async(process_sequence, (list(seq_paths2d),), callback=merge_examples)

    sorted_image_relpaths = sorted(image_to_examples.keys())
    examples = []
    all_detections = spu.load_pickle(f'{SAILVOS_ROOT}/yolov4_detections.pkl')

    # Now we do the association with detections, replace the ground truth boxes with detection boxes
    # and preprocess the ones that are worth using by precomputing a small resolution cropped image
    # per example
    with spu.ThrottledPool() as pool:
        for i, image_relpath in enumerate(spu.progressbar(sorted_image_relpaths)):
            try:
                p = image_relpath[len('sailvos/'):]
                detections = all_detections[p]
            except KeyError:
                print(f'No detection: {image_relpath}')
                continue

            gt_people = image_to_examples[image_relpath]
            iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                    for box in detections]
                                   for gt_person in gt_people])
            gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

            for i_gt, i_det in zip(gt_indices, det_indices):
                ex = gt_people[i_gt]
                det_box = np.array(detections[i_det][:4])
                enough_overlap_gt_and_det = iou_matrix[i_gt, i_det] > 0.1
                det_box_too_big = boxlib.area(det_box) > 2 * boxlib.area(ex.bbox)
                gt_box_big_enough = np.min(ex.bbox[2:]) > 50
                very_occluded = ex.occ_fraction > 0.75
                enough_joints_gt = np.count_nonzero(~np.isnan(ex.world_coords[..., 0])) > 5
                coords_sane = 100 < multidim_stdev(ex.world_coords) < 3000

                if (enough_overlap_gt_and_det and not very_occluded and not det_box_too_big and
                        gt_box_big_enough and enough_joints_gt and coords_sane):
                    ex.bbox = det_box
                    new_im_path = ex.image_path[:-4].replace('sailvos', 'sailvos_downscaled')
                    new_im_path = f'{new_im_path}_{i_gt:01d}.jpg'
                    pool.apply_async(
                        make_efficient_example, (ex, new_im_path), callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    examples_per_seq = spu.groupby(examples, lambda ex: get_seq_name(ex.image_path))
    for seq_name, seq_examples in examples_per_seq.items():
        mean_camera_position = np.mean([ex.camera.t for ex in seq_examples], axis=0)

        for ex in seq_examples:
            ex.camera.t -= mean_camera_position
            ex.world_coords -= mean_camera_position

    return ds3d.Pose3DDataset(joint_info, examples)


def process_sequence(filepaths2d):
    prev_coords = None
    image_to_examples = collections.defaultdict(list)
    for filepath2d in filepaths2d:
        camera = load_camera(filepath2d.replace('_2d.txt', '_camera.npz'))
        if camera is None:
            continue

        all_coords3d = np.loadtxt(filepath2d.replace('_2d.txt', '_3d.txt'), dtype=np.float32) * 1000
        all_coords3d[all_coords3d == 0] = np.nan
        world_coords = np.array([
            np.nanmean(all_coords3d[alts], axis=0) for alts in INTERESTING_JOINTS])

        if not sufficient_pose_change(prev_coords, world_coords):
            continue

        prev_coords = world_coords
        i_frame = int(osp.basename(filepath2d).split('_')[0])
        seq_name = get_seq_name(filepath2d)
        image_relpath = f'sailvos/{seq_name}/images/{i_frame:06d}.jpg'
        amodal_mask = imageio.imread(filepath2d.replace('_2d.txt', '.png')) / 255
        all_people_maskpaths = glob.glob(f'{SAILVOS_ROOT}/{seq_name}/*_Ped_*/{i_frame:06d}.png')
        visible_ids = np.load(f'{SAILVOS_ROOT}/{seq_name}/visible/{i_frame:06d}.npz')['visible_ids']
        object_id = get_object_id(filepath2d)
        all_people_ids = [get_object_id(p) for p in all_people_maskpaths]
        modal_mask_encoded = rlemasklib.encode(visible_ids == object_id)
        all_people_mask = rlemasklib.encode(
            np.any([visible_ids == i for i in all_people_ids], axis=0))
        bbox = rlemasklib.to_bbox(modal_mask_encoded)
        occ_frac = 1 - rlemasklib.area(modal_mask_encoded) / np.sum(amodal_mask)
        ex = ds3d.Pose3DExample(
            image_relpath, world_coords, bbox=bbox, camera=camera, mask=all_people_mask)
        ex.occ_fraction = occ_frac
        image_to_examples[image_relpath].append(ex)

    return image_to_examples


def get_seq_name(path):
    return path.split('/')[-3]


def get_object_id(path):
    return int(path.split('/')[-2].split('_')[0])


def load_camera(path):
    a = np.load(path)
    if np.all(np.isnan(a['intrinsics'])):
        return None

    cam = cameralib.Camera(extrinsic_matrix=a['extrinsics'], intrinsic_matrix=a['intrinsics'])
    cam.t *= 1000
    return cam


def sufficient_pose_change(prev_pose, current_pose):
    if prev_pose is None:
        return True
    valid_prev = ~np.any(np.isnan(prev_pose), axis=-1)
    valid_current = ~np.any(np.isnan(current_pose), axis=-1)
    valid = np.logical_and(valid_prev, valid_current)
    dists = np.linalg.norm(prev_pose - current_pose, axis=-1)
    return np.any(dists[valid] >= 100)


def multidim_stdev(x):
    mean = np.nanmean(x, axis=0)
    dist = np.linalg.norm(x - mean, axis=1)
    rms_dist = np.sqrt(np.nanmean(np.square(dist)))
    return rms_dist


INTERESTING_JOINTS = [
    [341, 8], [61], [216, 234], [171], [164, 309], [255, 39], [330], [108], [45, 294],
    [215, 131, 253], [295], [144, 315], [147, 342, 27], [85], [152], [206, 128, 102], [307, 312],
    [130], [11], [167, 188, 191, 310], [194, 193, 236, 78], [227, 304, 58, 229], [37, 260, 195],
    [47, 148, 232],
    [323], [160]
]

if __name__ == '__main__':
    make_dataset()
