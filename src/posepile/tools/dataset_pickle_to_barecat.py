import os.path as osp
import sys

import barecat
import cv2
import numpy as np
import posepile.datasets3d as ds3d
import rlemasklib
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.util import geom3d
from posepile.util.misc import cast_if_precise_enough


def convert(pickle_in_path, barecat_out_path, legacy_module_name):
    if legacy_module_name:
        import posepile.joint_info
        import posepile.datasets3d
        import posepile.datasets2d
        sys.modules[f"{legacy_module_name}.datasets3d"] = posepile.datasets3d
        sys.modules[f"{legacy_module_name}.datasets2d"] = posepile.datasets2d
        sys.modules[f"{legacy_module_name}.joint_info"] = posepile.joint_info
    ds = spu.load_pickle(pickle_in_path)
    fix_legacy_issues(ds)
    dataset_to_barecat(ds, barecat_out_path)


def update_pickle_file(pickle_in_path, pickle_out_path, legacy_module_name):
    if legacy_module_name:
        import posepile
        import posepile.joint_info
        import posepile.datasets3d
        import posepile.datasets2d
        sys.modules[f"{legacy_module_name}.datasets3d"] = posepile.datasets3d
        sys.modules[f"{legacy_module_name}.datasets2d"] = posepile.datasets2d
        sys.modules[f"{legacy_module_name}.joint_info"] = posepile.joint_info
        sys.modules[f"{legacy_module_name}"] = posepile
    ds = spu.load_pickle(pickle_in_path)
    fix_legacy_issues(ds)
    spu.dump_pickle(ds, pickle_out_path)


def dataset_to_barecat(ds, out_path):
    with barecat.Barecat(out_path, overwrite=True, readonly=False, auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=ds.joint_info.names,
            joint_edges=ds.joint_info.stick_figure_edges,
            train_bone_lengths=ds.train_bones,
            trainval_bone_lengths=ds.trainval_bones)

        for i_phase, phase_name in enumerate(['train', 'val', 'test']):
            examples = ds.examples[i_phase]
            examples_by_imagepath = spu.groupby(
                spu.progressbar(examples, desc='Grouping by image'),
                lambda ex: ex.image_path)

            for examples_of_image in spu.progressbar(
                    examples_by_imagepath.values(), total=len(examples_by_imagepath),
                    desc=phase_name):
                for i_example_in_image, ex in enumerate(examples_of_image):
                    stored_path = f'{phase_name}/{ex.image_path}_{i_example_in_image:02d}.msgpack'
                    bc_writer[stored_path] = example_to_dict(ex)


def get_joint_info(metadata):
    return JointInfo(metadata['joint_names'], metadata['joint_edges'])


def example_to_dict(ex):
    if isinstance(ex.world_coords, ds3d.SparseCoords):
        i_valid_joints = ex.world_coords.i_valid_joints
        valid_coords = ex.world_coords.valid_coords
    else:
        coords = ex.world_coords.astype(np.float32)
        i_valid_joints = np.where(geom3d.are_joints_valid(coords))[0].astype(np.uint16)
        valid_coords = np.ascontiguousarray(coords[i_valid_joints])
        valid_coords = cast_if_precise_enough(valid_coords, np.float16, threshold=1)

    if i_valid_joints.shape[0] == ex.world_coords.shape[0]:
        i_valid_joints = i_valid_joints[:0]

    result = dict(
        impath=ex.image_path,
        bbox=np.round(ex.bbox).astype(np.int16),
        joints3d=dict(
            rows=valid_coords,
            i_rows=np.asarray(i_valid_joints, np.uint16)
        ),
        cam=dict(
            rotvec_w2c=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            loc=ex.camera.t,
            intr=ex.camera.intrinsic_matrix[:2],
            up=ex.camera.world_up
        )
    )

    if (ex.camera.distortion_coeffs is not None and
            np.count_nonzero(ex.camera.distortion_coeffs) > 0):
        result['cam']['distcoef'] = ex.camera.distortion_coeffs

    if ex.mask is not None:
        result['mask'] = rlemasklib.compress(ex.mask, zlevel=-1)
    return result


def fix_legacy_issues(ds):
    def fix_path(p):
        if osp.isabs(p):
            return osp.normpath(osp.relpath(p, '/nodes/brewdog/work3/sarandi/data_reprod'))
        else:
            return osp.normpath(p)

    for ex in spu.progressbar(ds.iter_examples()):
        ex.image_path = fix_path(ex.image_path)
        if hasattr(ex, 'mask') and isinstance(ex.mask, list):
            assert len(ex.mask) == 1 or all(m == ex.mask[0] for m in ex.mask)
            ex.mask = ex.mask[0]
