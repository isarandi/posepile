import os.path as osp

import boxlib
import numpy as np
import posepile.datasets2d as ds2d
import scipy.optimize
import simplepyutils as spu
from posepile import util
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.util import TRAIN


@spu.picklecache('mpii.pkl', min_time="2020-11-04T16:46:25")
def make_dataset():
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,pelv,thor,neck,head,rwri,relb,rsho,lsho,lelb,lwri'
    edges = 'rsho-relb-rwri,rhip-rkne-rank,neck-head,pelv-thor'
    joint_info_full = JointInfo(joint_names, edges)

    joint_names_used = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'
    joint_info_used = JointInfo(joint_names_used, edges)
    dataset = ds2d.Pose2DDataset(joint_info_used)
    selected_joints = [joint_info_full.ids[name] for name in joint_info_used.names]

    mat_path = f'{DATA_ROOT}/mpii/mpii_human_pose_v1_u12_1.mat'
    s = util.load_mat(mat_path).RELEASE
    annolist = np.atleast_1d(s.annolist)
    with spu.ThrottledPool() as pool:
        for anno, is_train, rect_ids in zip(
                annolist, spu.progressbar(s.img_train), s.single_person):
            if not is_train:
                continue

            image_path = f'mpii/images/{anno.image.name}'
            annorect = np.atleast_1d(anno.annorect)
            rect_ids = np.atleast_1d(rect_ids) - 1

            for rect_id in rect_ids:
                rect = annorect[rect_id]
                if 'annopoints' not in rect or len(rect.annopoints) == 0:
                    continue

                coords = np.full(
                    shape=[joint_info_full.n_joints, 2], fill_value=np.nan, dtype=np.float32)
                for joint in np.atleast_1d(rect.annopoints.point):
                    coords[joint.id] = [joint.x, joint.y]

                coords = coords[selected_joints]
                rough_person_center = np.float32([rect.objpos.x, rect.objpos.y])
                rough_person_size = rect.scale * 200

                # Shift person center down like [Sun et al. 2018], who say this is common on MPII
                rough_person_center[1] += 0.075 * rough_person_size

                topleft = np.array(rough_person_center) - np.array(rough_person_size) / 2
                bbox = np.array([topleft[0], topleft[1], rough_person_size, rough_person_size])
                ex = ds2d.Pose2DExample(image_path, coords, bbox=bbox)
                new_im_path = image_path.replace('mpii', 'mpii_downscaled')
                without_ext, ext = osp.splitext(new_im_path)
                new_im_path = f'{without_ext}_{rect_id:02d}{ext}'
                pool.apply_async(
                    make_efficient_example, (ex, new_im_path),
                    callback=dataset.examples[TRAIN].append)

    dataset.examples[TRAIN].sort(key=lambda x: x.image_path)
    return dataset


@spu.picklecache('mpii_yolo.pkl', min_time="2021-06-01T21:39:35")
def make_mpii_yolo(filter_joints=True):
    joint_info_full = JointInfo(
        'rank,rkne,rhip,lhip,lkne,lank,pelv,thor,neck,head,rwri,relb,rsho,lsho,lelb,lwri',
        'lsho-lelb-lwri,rsho-relb-rwri,lhip-lkne-lank,rhip-rkne-rank,neck-head,pelv-thor')
    if filter_joints:
        joint_info_used = JointInfo(
            'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri',
            'lelb-lwri,relb-rwri,lhip-lkne-lank,rhip-rkne-rank')
        selected_joints = [joint_info_full.ids[name] for name in joint_info_used.names]
    else:
        joint_info_used = joint_info_full
        selected_joints = list(range(joint_info_used.n_joints))

    mat_path = f'{DATA_ROOT}/mpii/mpii_human_pose_v1_u12_1.mat'
    s = util.load_mat(mat_path).RELEASE
    annolist = np.atleast_1d(s.annolist)
    all_boxes = spu.load_pickle(f'{DATA_ROOT}/mpii/yolov3_detections.pkl')

    examples = []
    with spu.ThrottledPool() as pool:
        for anno_id, (anno, is_train) in enumerate(
                zip(annolist, spu.progressbar(s.img_train))):
            if not is_train:
                continue

            image_path = f'{DATA_ROOT}/mpii/images/{anno.image.name}'

            annorect = np.atleast_1d(anno.annorect)
            gt_people = []
            for rect_id, rect in enumerate(annorect):
                if 'annopoints' not in rect or len(rect.annopoints) == 0:
                    continue

                coords = np.full(
                    shape=[joint_info_full.n_joints, 2], fill_value=np.nan, dtype=np.float32)
                for joint in np.atleast_1d(rect.annopoints.point):
                    coords[joint.id] = [joint.x, joint.y]

                bbox = boxlib.expand(boxlib.bb_of_points(coords), 1.25)
                coords = coords[selected_joints]
                ex = ds2d.Pose2DExample(image_path, coords, bbox=bbox)
                gt_people.append(ex)

            if not gt_people:
                continue

            image_relpath = osp.relpath(f'images/{anno.image.name}')
            boxes = [box for box in all_boxes[image_relpath] if box[-1] > 0.5]
            if not boxes:
                continue

            iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                    for box in boxes]
                                   for gt_person in gt_people])
            gt_indices, box_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

            for i_gt, i_det in zip(gt_indices, box_indices):
                if iou_matrix[i_gt, i_det] > 0.1:
                    ex = gt_people[i_gt]
                    ex.bbox = np.array(boxes[i_det][:4])
                    suf = '' if filter_joints else '_nofilt'
                    new_im_path = image_path.replace('mpii', f'mpii_downscaled_yolo{suf}')
                    without_ext, ext = osp.splitext(new_im_path)
                    new_im_path = f'{without_ext}_{i_gt:02d}{ext}'
                    pool.apply_async(make_efficient_example, (ex, new_im_path),
                                     callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)

    def n_valid_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords), axis=-1))

    examples = [ex for ex in examples if n_valid_joints(ex) > 6]
    return ds2d.Pose2DDataset(joint_info_used, examples)
