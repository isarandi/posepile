import os.path as osp

import boxlib
import numpy as np
import posepile.datasets2d as ds2d
import pycocotools.coco
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util import TEST, TRAIN, VALID, improc
from posepile.ds.posetrack.main import get_coords
from posepile.ds.densepose_coco.main import make_efficient_example_dp, make_dataset as make_coco_dp


@spu.picklecache('densepose_posetrack.pkl', min_time="2023-11-14T03:24:48")
def make_dataset():
    joint_info = JointInfo(
        'nose,head,htop,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,rank',
        'rsho-relb-rwri,rhip-rkne-rank,nose-reye-rear,head-htop')
    n_joints = joint_info.n_joints
    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    body_joint_ids = [i for name, i in joint_info.ids.items()
                      if any(name.startswith(x) for x in body_joint_names)]

    def get_n_valid_body_joints(coords2d):
        return np.count_nonzero(np.all(~np.isnan(coords2d[body_joint_ids]), axis=-1))

    n_joints = joint_info.n_joints
    learning_phase_shortnames = {TRAIN: 'train', VALID: 'val', TEST: 'test'}
    examples_per_phase = {TRAIN: [], VALID: []}

    with (spu.ThrottledPool() as pool):
        for example_phase in (TRAIN,):
            phase_shortname = learning_phase_shortnames[example_phase]
            coco = pycocotools.coco.COCO(
                f'{DATA_ROOT}/densepose/DensePose_PoseTrack/'
                f'densepose_only_posetrack_{phase_shortname}2017.json')

            for ann in spu.progressbar(coco.anns.values(), total=len(coco.anns)):
                if not 'dp_masks' in ann.keys():
                    continue
                bbox = np.round(ann['bbox'])
                filename = coco.imgs[ann['image_id']]['file_name']
                frame_id = coco.imgs[ann['image_id']]['frame_id']

                basename = osp.basename(filename)
                base_noext, ext = osp.splitext(basename)

                dirname = spu.split_path(filename)[-2]
                seq_num, institute = dirname.split('_')
                seq_num = int(seq_num)
                image_path = (
                    f'{DATA_ROOT}/posetrack/images/{phase_shortname}/'
                    f'{seq_num:06d}_{institute}_{phase_shortname}/{frame_id:06d}.jpg')
                if not osp.exists(image_path):
                    print('not found', image_path)
                    continue

                dense2d = np.stack([ann['dp_x'], ann['dp_y']], axis=1)
                if len(dense2d) == 0:
                    continue

                coords = get_coords(ann)
                width = coco.imgs[ann['image_id']]['width']
                height = coco.imgs[ann['image_id']]['height']
                bbox_cropped = boxlib.intersection(bbox, boxlib.full(imsize=(width, height)))
                dense2d = dense2d / 255. * bbox_cropped[2:] + bbox_cropped[:2]
                dense2d = dense2d.astype(np.float32)

                bbox_pt1 = np.array(ann['bbox'][:2], np.float32)
                bbox_wh = np.array(ann['bbox'][2:4], np.float32)
                bbox = np.array([*bbox_pt1, *bbox_wh], np.float32)
                im_relpath = osp.relpath(image_path, DATA_ROOT)
                ex = ds2d.Pose2DExample(
                    im_relpath, coords, bbox=bbox,
                    densepose=(dense2d, (ann['dp_I'], ann['dp_U'], ann['dp_V'])))
                new_im_relpath = im_relpath.replace(
                    'posetrack', 'densepose_posetrack_downscaled')
                without_ext, ext = osp.splitext(new_im_relpath)
                i_example = ann['id']
                new_im_relpath = f'{without_ext}_{i_example}{ext}'
                pool.apply_async(
                    make_efficient_example_dp, (ex, new_im_relpath),
                    callback=examples_per_phase[example_phase].append)

    examples_per_phase[TRAIN].sort(key=lambda ex: ex.image_path)
    return ds2d.Pose2DDataset(joint_info, examples_per_phase[TRAIN])


@spu.picklecache('densepose_posetrack_like_coco.pkl', min_time="2023-11-14T03:24:48")
def make_like_coco():
    coco_joint_info = make_coco_dp().joint_info
    posetrack_dp = make_dataset()
    for ex in posetrack_dp.iter_examples():
        ex.coords[1:3] = np.nan

    posetrack_dp.joint_info = coco_joint_info
    return posetrack_dp


if __name__ == '__main__':
    make_dataset()
