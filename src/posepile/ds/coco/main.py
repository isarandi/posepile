import os.path as osp

import boxlib
import numpy as np
import pycocotools.coco
import simplepyutils as spu

import posepile.datasets2d as ds2d
import posepile.joint_filtering
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example

from posepile.util import TEST, TRAIN, VALID
import argparse
from simplepyutils import FLAGS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wholebody', action=spu.argparse.BoolAction)
    spu.initialize(parser)

    if FLAGS.wholebody:
        make_wholebody()
    else:
        make_dataset()

@spu.picklecache('coco.pkl', min_time="2020-02-01T02:53:21")
def make_dataset(single_person=True):
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear')
    n_joints = joint_info.n_joints
    learning_phase_shortnames = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2
    iou_threshold = 0.1 if single_person else 0.5

    suffix = '' if single_person else '_multi'
    examples_per_phase = {TRAIN: [], VALID: []}
    with spu.ThrottledPool() as pool:
        for example_phase in (TRAIN, VALID):
            phase_shortname = learning_phase_shortnames[example_phase]
            coco_filepath = (
                f'{DATA_ROOT}/coco/annotations/person_keypoints_{phase_shortname}2014.json')
            coco = pycocotools.coco.COCO(coco_filepath)

            impath_to_examples = {}
            for ann in coco.anns.values():
                filename = coco.imgs[ann['image_id']]['file_name']
                image_path = f'{DATA_ROOT}/coco/{phase_shortname}2014/{filename}'

                joints = np.array(ann['keypoints']).reshape([-1, 3])
                visibilities = joints[:, 2]
                coords = joints[:, :2].astype(np.float32).copy()
                n_visible_joints = np.count_nonzero(visibilities == VISIBLE)
                n_occluded_joints = np.count_nonzero(visibilities == OCCLUDED)
                n_labeled_joints = n_occluded_joints + n_visible_joints

                if n_visible_joints >= n_joints / 3 and n_labeled_joints >= n_joints / 2:
                    coords[visibilities == UNLABELED] = np.nan
                    bbox_pt1 = np.array(ann['bbox'][:2], np.float32)
                    bbox_wh = np.array(ann['bbox'][2:4], np.float32)
                    bbox = np.array([*bbox_pt1, *bbox_wh])
                    mask = coco.annToRLE(ann)
                    ex = ds2d.Pose2DExample(image_path, coords, bbox=bbox, mask=mask)
                    impath_to_examples.setdefault(image_path, []).append(ex)

            n_images = len(impath_to_examples)
            for impath, examples in spu.progressbar_items(impath_to_examples):
                for i_example, example in enumerate(examples):
                    box = boxlib.expand(boxlib.bb_of_points(example.coords), 1.25)
                    if np.max(box[2:]) < 200:
                        continue

                    if single_person:
                        other_boxes = [boxlib.expand(boxlib.bb_of_points(e.coords), 1.25)
                                       for e in examples if e is not example]
                        ious = np.array([boxlib.iou(b, box) for b in other_boxes])
                        usable = np.all(ious < iou_threshold)
                    else:
                        usable = True

                    if usable:
                        new_im_path = impath.replace('coco', 'coco_downscaled' + suffix)
                        without_ext, ext = osp.splitext(new_im_path)
                        new_im_path = f'{without_ext}_{i_example:02d}{ext}'
                        new_im_relpath = osp.relpath(new_im_path, DATA_ROOT)
                        pool.apply_async(
                            make_efficient_example, (example, new_im_relpath),
                            callback=examples_per_phase[example_phase].append)

    examples_per_phase[TRAIN].sort(key=lambda ex: ex.image_path)
    examples_per_phase[VALID].sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples_per_phase[TRAIN], examples_per_phase[VALID])


@spu.picklecache('coco_wb.pkl', min_time="2020-02-01T02:53:21")
def make_wholebody():
    import xtcocotools.coco
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,'
        'rank,ltoe,ltoe2,lhee,rtoe,rtoe2,rhee,rface1,rface2,rface3,rface4,rface5,rface6,rface7,'
        'rface8,chin,'
        'lface8,lface7,lface6,lface5,lface4,lface3,lface2,lface1,rface9,rface10,rface11,rface12,'
        'rface13,lface13,lface12,lface11,lface10,'
        'lface9,nose1,nose2,nose3,nose4,rface14,rface15,nose5,lface15,lface14,reye1,reye2,'
        'reye3,reye4,reye5,reye6,leye4,leye3,leye2,leye1,leye6,leye5,rmouth1,rmouth2,'
        'rmouth3,mouth1,lmouth3,lmouth2,lmouth1,lmouth4,lmouth5,mouth2,rmouth5,rmouth4,'
        'rmouth6,rmouth7,mouth3,lmouth7,lmouth6,lmouth8,mouth4,rmouth8,lhan1,lhan2,lhan3,'
        'lhan4,lhan5,lhan6,lhan7,lhan8,lhan9,lhan10,lhan11,lhan12,lhan13,lhan14,lhan15,'
        'lhan16,lhan17,lhan18,lhan19,lhan20,lhan21,rhan1,rhan2,rhan3,rhan4,rhan5,rhan6,'
        'rhan7,rhan8,rhan9,rhan10,rhan11,rhan12,rhan13,rhan14,rhan15,rhan16,rhan17,rhan18,'
        'rhan19,rhan20,rhan21',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear')

    n_joints = joint_info.n_joints
    n_body_joints = 17

    learning_phase_shortnames = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2

    examples_per_phase = {TRAIN: [], VALID: []}
    with spu.ThrottledPool() as pool:
        for example_phase in (TRAIN, VALID):
            phase_shortname = learning_phase_shortnames[example_phase]
            coco_filepath = (
                f'{DATA_ROOT}/coco_wb/coco_wholebody_{phase_shortname}_v1.0.json')
            coco = xtcocotools.coco.COCO(coco_filepath)

            impath_to_examples = {}
            for _, ann in spu.progressbar_items(coco.anns):
                filename = coco.imgs[ann['image_id']]['file_name']
                image_path = f'{DATA_ROOT}/coco/{phase_shortname}2017/{filename}'

                body_joints = np.array(ann['keypoints']).reshape([-1, 3])
                visibilities = body_joints[:, 2]
                body_coords = body_joints[:, :2].astype(np.float32).copy()
                n_visible_joints = np.count_nonzero(visibilities == VISIBLE)
                n_occluded_joints = np.count_nonzero(visibilities == OCCLUDED)
                n_labeled_joints = n_occluded_joints + n_visible_joints

                coords_parts = [body_coords]

                if n_visible_joints >= n_body_joints / 3 and n_labeled_joints >= n_body_joints / 2:
                    body_coords[visibilities == UNLABELED] = np.nan

                    for othername in ['foot', 'face', 'lefthand', 'righthand']:
                        other_joints = np.array(ann[f'{othername}_kpts']).reshape([-1, 3])
                        if ann[f'{othername}_valid']:
                            other_coords = other_joints[:, :2].astype(np.float32)
                            other_coords[other_joints[:, 2] <= 0] = np.nan
                            coords_parts.append(other_coords)
                        else:
                            coords_parts.append(np.full_like(other_joints[:, :2], np.nan))
                    coords = np.concatenate(coords_parts, axis=0)
                    bbox_pt1 = np.array(ann['bbox'][:2], np.float32)
                    bbox_wh = np.array(ann['bbox'][2:4], np.float32)
                    bbox = np.array([*bbox_pt1, *bbox_wh])
                    mask = coco.annToRLE(ann)
                    ex = ds2d.Pose2DExample(image_path, coords, bbox=bbox, mask=mask)
                    impath_to_examples.setdefault(image_path, []).append(ex)

            n_images = len(impath_to_examples)
            for impath, examples in spu.progressbar_items(impath_to_examples):
                for i_example, example in enumerate(examples):
                    box = boxlib.expand(boxlib.bb_of_points(example.coords), 1.25)
                    if np.max(box[2:]) < 200:
                        continue

                    new_im_path = impath.replace('coco', 'coco_wb_downscaled')
                    without_ext, ext = osp.splitext(new_im_path)
                    new_im_path = f'{without_ext}_{i_example:02d}{ext}'
                    new_im_relpath = osp.relpath(new_im_path, DATA_ROOT)
                    pool.apply_async(
                        make_efficient_example, (example, new_im_relpath),
                        callback=examples_per_phase[example_phase].append)

    examples_per_phase[TRAIN].sort(key=lambda ex: ex.image_path)
    examples_per_phase[VALID].sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples_per_phase[TRAIN], examples_per_phase[VALID])


@spu.picklecache('coco_reduced.pkl', min_time="2020-02-01T02:53:21")
def make_reduced_dataset(single_person=False, face=True):
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'
    if face:
        joint_names += ',nose,leye,reye,lear,rear'

    edges = 'lelb-lwri,relb-rwri,lhip-lkne-lank,rhip-rkne-rank'
    joint_info = JointInfo(joint_names, edges)
    ds = posepile.joint_filtering.convert_dataset(make_dataset(single_person), joint_info)

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'.split(',')
    body_joint_ids = [joint_info.ids[name] for name in body_joint_names]

    def n_valid_body_joints(example):
        return np.count_nonzero(
            np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    ds.examples[TRAIN] = [ex for ex in ds.examples[TRAIN] if n_valid_body_joints(ex) > 6]
    return ds



if __name__ == '__main__':
    main()
