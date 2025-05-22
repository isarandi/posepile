import glob
import os.path as osp

import boxlib
import numpy as np
import scipy.optimize
import simplepyutils as spu

import posepile.datasets2d as ds2d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('posetrack.pkl', min_time="2019-11-28T01:12:12")
def make_dataset():
    root = f'{DATA_ROOT}/posetrack'
    all_detections = spu.load_pickle(f'{root}/yolov4_detections.pkl')
    anno_paths = glob.glob(f'{root}/posetrack_data/annotations/*/*.json')
    anno_paths = [x for x in anno_paths if 'posetrack_data/annotations/test/' not in x
                  and 'preds_val' not in x]
    examples_train = []
    examples_val = []
    with spu.ThrottledPool() as pool:
        for anno_path in spu.progressbar(anno_paths):
            is_train = 'posetrack_data/annotations/train/' in anno_path
            try:
                info = spu.load_json(anno_path)
            except:
                print(anno_path)
                raise
            annos_per_image = spu.groupby(info['annotations'], lambda x: x['image_id'])
            image_id_to_relpath = {}
            for img_info in info['images']:
                image_id_to_relpath[img_info['id']] = img_info['file_name']

            prev_pose_per_track = {}
            for image_id in sorted(annos_per_image):
                annos = annos_per_image[image_id]
                annos = [x for x in annos if 'bbox' in x]
                if not annos:
                    continue

                image_path = f'{root}/{image_id_to_relpath[image_id]}'
                detections = all_detections[osp.relpath(image_path, f'{root}/images')]
                iou_matrix = np.array([[boxlib.iou(anno['bbox'], det[:4])
                                        for det in detections]
                                       for anno in annos])

                gt_indices, box_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
                for i_gt, i_det in zip(gt_indices, box_indices):
                    if iou_matrix[i_gt, i_det] < 0.1:
                        continue

                    anno = annos[i_gt]
                    coords = get_coords(anno)
                    is_valid = np.logical_not(np.isnan(coords[:, 0]))
                    track_id = anno['track_id']

                    if track_id in prev_pose_per_track:
                        prev_pose = prev_pose_per_track[track_id]
                        prev_is_valid = np.logical_not(np.isnan(prev_pose[:, 0]))
                        newly_valid = np.logical_and(np.logical_not(prev_is_valid), is_valid)
                        head_box = anno['bbox_head']
                        movement_thresh = np.linalg.norm(np.hypot(head_box[2], head_box[3])) * 0.5

                        # BUG!! fix to axis=-1
                        change = np.linalg.norm(coords[is_valid, :2] - prev_pose[is_valid, :2])
                        if not np.any(newly_valid) and np.all(change < movement_thresh):
                            continue

                    prev_pose_per_track[track_id] = coords
                    image_relpath = osp.relpath(image_path, DATA_ROOT)
                    new_im_relpath = image_relpath.replace('posetrack', 'posetrack_downscaled')
                    without_ext, ext = osp.splitext(new_im_relpath)
                    new_im_relpath = f'{without_ext}_{i_gt:02d}{ext}'
                    ex = ds2d.Pose2DExample(image_relpath, coords, bbox=detections[i_det, :4])

                    example_container = examples_train if is_train else examples_val
                    pool.apply_async(make_efficient_example, (ex, new_im_relpath),
                                     callback=example_container.append)

    joint_info = JointInfo(
        'nose,head,htop,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear,head-htop')
    examples_train.sort(key=lambda ex: ex.image_path)
    examples_val.sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples_train, examples_val)


def get_coords(anno):
    coords = np.array(anno['keypoints'], np.float32).reshape([-1, 3])
    is_valid = coords[:, 2] == 1
    coords = coords[:, :2]
    coords[np.logical_not(is_valid)] = np.nan
    return coords


if __name__ == '__main__':
    make_dataset()
