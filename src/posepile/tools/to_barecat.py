import argparse

import barecat
import cv2
import numpy as np
import rlemasklib
import simplepyutils as spu
from posepile.util import geom3d
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    spu.initialize(parser)
    ds = spu.load_pickle(FLAGS.input_path)
    metadata = dict(
        joint_names=ds.joint_info.names,
        joint_edges=ds.joint_info.stick_figure_edges)
    if hasattr(ds, 'train_bones'):
        metadata['train_bone_lengths'] = ds.train_bones
    if hasattr(ds, 'trainval_bones'):
        metadata['trainval_bone_lengths'] = ds.trainval_bones

    with barecat.Barecat(
            FLAGS.output_path, overwrite=True, auto_codec=True,
            readonly=False) as bc_writer:
        bc_writer['metadata.msgpack'] = metadata

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


def example_to_dict(ex):
    result = dict(
        impath=ex.image_path,
        bbox=np.round(ex.bbox).astype(np.int16),
        cam=dict(
            rotvec_w2c=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            loc=ex.camera.t,
            intr=ex.camera.intrinsic_matrix[:2],
            up=ex.camera.world_up
        )
    )

    if ex.parameters is not None:
        parameters = {**ex.parameters}
        if 'kid_factor' in parameters and parameters['kid_factor'] == 0:
            del parameters['kid_factor']
        if 'scale' in parameters and parameters['scale'] == 1:
            del parameters['scale']
        if 'expression' in parameters and np.all(parameters['expression'] == 0):
            del parameters['expression']
        result['parameters'] = parameters

    if ex.world_coords is not None:
        coords = ex.world_coords.astype(np.float32)
        i_valid_joints = np.where(geom3d.are_joints_valid(coords))[0].astype(np.uint16)
        valid_coords = np.ascontiguousarray(coords[i_valid_joints])
        valid_coords_fp16 = valid_coords.astype(np.float16)
        diff = np.nanmax(np.abs(valid_coords - valid_coords_fp16))
        if diff < 1:
            valid_coords = valid_coords_fp16

        if i_valid_joints.shape[0] == ex.world_coords.shape[0]:
            i_valid_joints = i_valid_joints[:0]
        result['joints3d'] = dict(rows=valid_coords, i_rows=np.asarray(i_valid_joints, np.uint16))

    if (ex.camera.distortion_coeffs is not None and
            np.count_nonzero(ex.camera.distortion_coeffs) > 0):
        result['cam']['distcoef'] = ex.camera.distortion_coeffs

    if ex.mask is not None:
        result['mask'] = rlemasklib.compress(ex.mask, zlevel=-1)
    return result


if __name__ == '__main__':
    main()
