import argparse
import os.path as osp

import boxlib
import numpy as np
import posepile.datasets2d as ds2d
import posepile.ds.densepose_coco.util as densepose_util
import pycocotools.coco
import rlemasklib
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util import TEST, TRAIN, VALID, improc
from posepile.util.preproc_for_efficiency import make_efficient_example


def main():
    parser = argparse.ArgumentParser()
    spu.initialize(parser)
    make_dataset()


@spu.picklecache('densepose_coco.pkl', min_time="2023-10-15T23:30:06")
def make_dataset():
    joint_info = JointInfo(
        'nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear',
    )

    body_joint_names = tuple(
        'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    )
    body_joint_ids = [i for name, i in joint_info.ids.items() if name.startswith(body_joint_names)]

    def get_n_valid_body_joints(coords2d):
        return np.count_nonzero(np.all(~np.isnan(coords2d[body_joint_ids]), axis=-1))

    n_joints = joint_info.n_joints
    learning_phase_shortnames = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
    examples_per_phase = {TRAIN: [], VALID: []}
    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2

    with spu.ThrottledPool() as pool:
        for example_phase in (TRAIN,):
            phase_shortname = learning_phase_shortnames[example_phase]
            coco = pycocotools.coco.COCO(
                f'{DATA_ROOT}/densepose/DensePose_COCO/densepose_coco_2014_{phase_shortname}.json'
            )

            for ann in spu.progressbar(coco.anns.values(), total=len(coco.anns)):
                if not 'dp_masks' in ann.keys():
                    continue
                bbox = np.round(ann['bbox'])
                filename = coco.imgs[ann['image_id']]['file_name']
                image_path = f'{DATA_ROOT}/coco/{phase_shortname}2014/{filename}'

                dense2d = np.stack([ann['dp_x'], ann['dp_y']], axis=1)
                if len(dense2d) == 0:
                    continue

                joints = np.array(ann['keypoints']).reshape([-1, 3])
                visibilities = joints[:, 2]
                coords = joints[:, :2].astype(np.float32).copy()
                n_visible_joints = np.count_nonzero(visibilities == VISIBLE)
                n_occluded_joints = np.count_nonzero(visibilities == OCCLUDED)
                n_labeled_joints = n_occluded_joints + n_visible_joints
                coords[visibilities == UNLABELED] = np.nan
                n_valid_body_joints = get_n_valid_body_joints(coords)

                if (
                    n_visible_joints < n_joints / 3
                    or n_labeled_joints < n_joints / 2
                    or n_valid_body_joints <= 6
                ):
                    continue

                width, height = improc.image_extents(image_path)
                bbox_cropped = boxlib.intersection(bbox, boxlib.full(imsize=(width, height)))
                dense2d = dense2d / 255.0 * bbox_cropped[2:] + bbox_cropped[:2]
                dense2d = dense2d.astype(np.float32)

                bbox_pt1 = np.array(ann['bbox'][:2], np.float32)
                bbox_wh = np.array(ann['bbox'][2:4], np.float32)
                bbox = np.array([*bbox_pt1, *bbox_wh], np.float32)
                mask = coco.annToRLE(ann)
                im_relpath = osp.relpath(image_path, DATA_ROOT)
                ex = ds2d.Pose2DExample(
                    im_relpath,
                    coords,
                    bbox=bbox,
                    mask=mask,
                    densepose=(dense2d, (ann['dp_I'], ann['dp_U'], ann['dp_V'])),
                )
                new_im_relpath = im_relpath.replace('coco', 'densepose_coco_downscaled')
                without_ext, ext = osp.splitext(new_im_relpath)
                i_example = ann['id']
                new_im_relpath = f'{without_ext}_{i_example}{ext}'
                pool.apply_async(
                    make_efficient_example_dp,
                    (ex, new_im_relpath),
                    callback=examples_per_phase[example_phase].append,
                )

    examples_per_phase[TRAIN].sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples_per_phase[TRAIN])


def make_efficient_example_dp(ex, new_im_path):
    dense2d, (i, u, v) = ex.densepose
    faces, barycoords = densepose_util.get_iuv_converter()(i, u, v)
    ex.densepose = (dense2d, faces, barycoords)
    return make_efficient_example(ex, new_im_path)


def example_to_dict(ex):
    result = dict(
        impath=ex.image_path,
        bbox=np.round(ex.bbox).astype(np.int16),
        parameters=ex.parameters,
        cam=dict(
            rotvec_w2c=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            loc=ex.camera.t,
            intr=ex.camera.intrinsic_matrix[:2],
            up=ex.camera.world_up,
        ),
    )
    if (
        ex.camera.distortion_coeffs is not None
        and np.count_nonzero(ex.camera.distortion_coeffs) > 0
    ):
        result['cam']['distcoef'] = ex.camera.distortion_coeffs

    if ex.mask is not None:
        result['mask'] = rlemasklib.compress(ex.mask, zlevel=-1)
    return result


if __name__ == '__main__':
    main()
