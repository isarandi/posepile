import argparse

import barecat
import numpy as np
import rlemasklib
import posepile.datasets2d as ds2d
import simplepyutils as spu
from posepile import joint_filtering
from posepile.joint_info import JointInfo
from posepile.merging.merged_dataset3d import merge_joint_infos_of_datasets
from posepile.util import TEST, TRAIN, VALID, geom3d
from simplepyutils import FLAGS
import cv2
from posepile.util.misc import cast_if_precise_enough


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--out-path', type=str)
    spu.initialize(parser)

    if FLAGS.name == 'huge2d':
        make_huge2d()
    elif FLAGS.name == 'huge2d2':
        make_huge2d2(FLAGS.out_path)
    elif FLAGS.name == 'dense_multi':
        make_dense_multi(FLAGS.out_path)
    else:
        raise ValueError()


@spu.picklecache('huge2d.pkl', min_time="2022-08-06T14:30:53")
def make_huge2d(separate=True, no_torso=False, no_face=False):
    import posepile.ds.mpii.main

    mpii = posepile.ds.mpii.main.make_mpii_yolo(filter_joints=False)

    coco = ds2d.get_dataset('coco', single_person=False)
    posetrack = ds2d.get_dataset('posetrack')
    jrdb2d = ds2d.get_dataset('jrdb')

    print('Merging...')
    ds = merge_datasets(
        [
            [mpii, [(0, 0)], 'mpii' if separate else ''],
            [coco, [(0, 0), (1, 0)], 'coco' if separate else ''],
            [posetrack, [(0, 0)], 'posetrack' if separate else ''],
            [jrdb2d, [(0, 0)], 'jrdb' if separate else ''],
        ]
    )
    print('Training set size:', len(ds.examples[0]))

    body_joint_names = tuple(
        'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    )
    body_joint_ids = [
        i for name, i in ds.joint_info.ids.items() if name.startswith(body_joint_names)
    ]

    def n_valid_body_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    ds.examples[TRAIN] = [ex for ex in ds.examples[TRAIN] if n_valid_body_joints(ex) > 6]

    if not separate:
        if no_torso:
            joint_info_used = JointInfo(
                'rank,rkne,lkne,lank,rwri,relb,lelb,lwri,leye,reye,lear,rear,nose',
                'relb-rwri,rkne-rank,rear-reye-nose',
            )
        elif no_face:
            joint_info_used = JointInfo(
                'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri',
                'rsho-relb-rwri,rhip-rkne-rank',
            )
        else:
            joint_info_used = JointInfo(
                'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri,leye,reye,lear,rear,'
                'nose',
                'rsho-relb-rwri,rhip-rkne-rank,rear-reye-nose',
            )
        return joint_filtering.convert_dataset(ds, joint_info_used)

    return ds


# @spu.picklecache('huge2d2.pkl', min_time="2024-02-18T01:13:38")
def make_huge2d2(bc_out_path, separate=True):
    import posepile.ds.mpii.main

    mpii = posepile.ds.mpii.main.make_mpii_yolo(filter_joints=False)

    coco = ds2d.make_coco_wholebody()
    posetrack = ds2d.get_dataset('posetrack')
    jrdb2d = ds2d.get_dataset('jrdb')
    aic = ds2d.get_dataset('aic')
    halpe = ds2d.get_dataset('halpe')

    print('Merging...')
    ds = merge_datasets(
        [
            [mpii, [(0, 0)], 'mpii' if separate else ''],
            [coco, [(0, 0)], 'coco' if separate else ''],
            [posetrack, [(0, 0)], 'posetrack' if separate else ''],
            [jrdb2d, [(0, 0)], 'jrdb' if separate else ''],
            [aic, [(0, 0)], 'aic' if separate else ''],
            [halpe, [(0, 0)], 'halpe' if separate else ''],
        ]
    )
    print('Training set size:', len(ds.examples[0]))

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri'.split(',')
    body_joint_ids = [
        i
        for name, i in ds.joint_info.ids.items()
        if any(name.startswith(x) for x in body_joint_names)
    ]

    def n_valid_body_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    ds.examples[TRAIN] = [ex for ex in ds.examples[TRAIN] if n_valid_body_joints(ex) > 6]

    if not separate:
        if no_torso:
            joint_info_used = JointInfo(
                'rank,rkne,lkne,lank,rwri,relb,lelb,lwri,leye,reye,lear,rear,nose',
                'relb-rwri,rkne-rank,rear-reye-nose',
            )
        elif no_face:
            joint_info_used = JointInfo(
                'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri',
                'rsho-relb-rwri,rhip-rkne-rank',
            )
        else:
            joint_info_used = JointInfo(
                'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri,leye,reye,lear,rear,'
                'nose',
                'rsho-relb-rwri,rhip-rkne-rank,rear-reye-nose',
            )
        ds = joint_filtering.convert_dataset(ds, joint_info_used)

    write_dataset_as_barecat(bc_out_path, ds)


def write_dataset_as_barecat(bc_out_path, ds):
    with barecat.Barecat(
        bc_out_path, readonly=False, overwrite=True, auto_codec=True
    ) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=ds.joint_info.names, joint_edges=ds.joint_info.stick_figure_edges
        )

        for i_phase, phase_name in enumerate(['train', 'val', 'test']):
            examples = ds.examples[i_phase]
            examples_by_imagepath = spu.groupby(
                spu.progressbar(examples, desc='Grouping by image'), lambda ex: ex.image_path
            )

            for examples_of_image in spu.progressbar(
                examples_by_imagepath.values(), total=len(examples_by_imagepath), desc=phase_name
            ):
                for i_example_in_image, ex in enumerate(examples_of_image):
                    stored_path = f'{phase_name}/{ex.image_path}_{i_example_in_image:02d}.msgpack'
                    bc_writer[stored_path] = example_to_dict(ex)


def merge_datasets(datasets_with_uses_suf):
    merged_joint_info = merge_joint_infos_of_datasets(datasets_with_uses_suf)

    for i_ds in range(len(datasets_with_uses_suf)):
        datasets_with_uses_suf[i_ds][0] = joint_filtering.convert_dataset(
            datasets_with_uses_suf[i_ds][0], merged_joint_info, update_bones=False
        )

    examples = {TRAIN: [], VALID: [], TEST: []}
    for ds, uses, suf in datasets_with_uses_suf:
        for take, use_as in uses:
            examples[use_as] += ds.examples[take]

    return ds2d.Pose2DDataset(merged_joint_info, examples[0], examples[1], examples[2])


def make_dense_multi(bc_out_path):
    coco = ds2d.get_dataset('densepose_coco')
    posetrack = ds2d.get_dataset('densepose_posetrack')
    merged_dataset = merge_datasets(
        [
            [coco, [(0, 0)], 'coco'],
            [posetrack, [(0, 0)], 'posetrack'],
        ]
    )
    write_dataset_as_barecat(bc_out_path, merged_dataset)


def example_to_dict(ex):
    coords = ex.coords.astype(np.float32)
    i_valid_joints = np.where(geom3d.are_joints_valid(coords))[0].astype(np.uint16)
    valid_coords = np.ascontiguousarray(coords[i_valid_joints])
    valid_coords = cast_if_precise_enough(valid_coords, np.float16, threshold=1)

    if i_valid_joints.shape[0] == ex.coords.shape[0]:
        i_valid_joints = i_valid_joints[:0]

    result = dict(
        impath=ex.image_path,
        bbox=np.round(ex.bbox).astype(np.int16),
        joints2d=dict(rows=valid_coords, i_rows=np.asarray(i_valid_joints, np.uint16)),
    )

    if hasattr(ex, 'camera') and ex.camera is not None:
        result['cam'] = dict(
            rotvec_w2c=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            loc=ex.camera.t,
            intr=ex.camera.intrinsic_matrix[:2],
            up=ex.camera.world_up,
        )

        if (
            ex.camera.distortion_coeffs is not None
            and np.count_nonzero(ex.camera.distortion_coeffs) > 0
        ):
            result['cam']['distcoef'] = ex.camera.distortion_coeffs

    if hasattr(ex, 'mask') and ex.mask is not None:
        result['mask'] = rlemasklib.compress(ex.mask, zlevel=-1)

    if hasattr(ex, 'densepose') and ex.densepose is not None:
        (dense2d, faces, barycoords) = ex.densepose

        result['densepose'] = dict(
            imcoords=cast_if_precise_enough(dense2d, np.float16, threshold=1),
            i_faces=np.array(faces, dtype=np.uint16),
            barycoords=cast_if_precise_enough(barycoords, np.float16, threshold=1e-2),
        )

    return result


if __name__ == '__main__':
    main()
