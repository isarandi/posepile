import argparse
import zlib

import h5py
import numpy as np
import posepile.util.geom3d as geom3d
import posepile.util.improc as improc
from spu import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-pickle-file', type=str)
    parser.add_argument('--output-hdf5-file', type=str)
    spu.initialize(parser)

    ds = spu.load_pickle(FLAGS.input_pickle_file)
    with h5py.File(FLAGS.output_hdf5_file, 'w') as f:
        pickle_to_hdf5(ds, f)


def pickle_to_hdf5(ds, f):
    joint_info_group = f.create_group(name='joint_info')
    joint_info_group.create_dataset(name='joint_names', data=ds.joint_info.names)
    joint_info_group.create_dataset(name='joint_names', data=ds.joint_info.names)
    joint_info_group.create_dataset(name='joint_edges', data=ds.joint_info.stick_figure_edges)
    joint_info_group.create_dataset(name='train_bone_lengths', data=ds.train_bones)
    joint_info_group.create_dataset(name='trainval_bone_lengths', data=ds.trainval_bones)

    compound_dtype = make_compound_dtype(dict(
        image_path=h5py.string_dtype(encoding='utf-8'),
        imshape=(np.uint16, (2,)),
        mask_zipped_runlengths=h5py.vlen_dtype(np.uint8),
        bbox=(np.int16, (4,)),
        pose3d_world=dict(
            coords_fp32=h5py.vlen_dtype((np.float32, (3,))),
            coords_fp16=h5py.vlen_dtype((np.float16, (3,))),
            row_indices=h5py.vlen_dtype((np.uint16, (3,))),
        ),
        camera=dict(
            rotvec_world_to_camera=(np.float32, (3,)),
            optical_center=(np.float32, (3,)),
            intrinsic_matrix=(np.float32, (2, 3)),
            distortion_coeffs=h5py.vlen_dtype(np.float32),
        )
    ))

    for phase_name, examples in zip(['train', 'val', 'test'], ds.examples):
        phase_ds = f.create_dataset(name=phase_name, shape=(len(examples),), dtype=compound_dtype)
        for i, ex in enumerate(examples):
            phase_ds[i] = make_compound_value(example_to_hdf5(ex), compound_dtype)


def make_compound_dtype(d):
    if isinstance(d, dict):
        dtype_list = []
        for key, value in d.items():
            if isinstance(value, dict):
                dtype_list.append((key, make_compound_dtype(value)))
            else:
                dtype_list.append((key, value))
        return np.dtype(dtype_list)
    else:
        return np.dtype(d)

def make_compound_value(d, dtype):
    if dtype.names is None:
        return d
    return np.asarray(tuple([
        make_compound_value(d[name], dtype[name])
        for name in dtype.names]), dtype)


def example_to_hdf5(ex):
    if isinstance(ex.world_coords, posepile.datasets3d.SparseCoords):
        i_valid_joints = ex.world_coords.i_valid_joints
        valid_coords = ex.world_coords.valid_coords
        if valid_coords.dtype == np.float32:
            valid_coords_fp16 = valid_coords.astype(np.float16)
            valid_coords = valid_coords[:0]
        else:
            valid_coords_fp16 = valid_coords
            valid_coords = valid_coords[:0].astype(np.float32)
    else:
        coords = ex.world_coords.astype(np.float32)
        i_valid_joints = np.where(geom3d.are_joints_valid(coords))[0].astype(np.uint16)
        valid_coords = np.ascontiguousarray(coords[i_valid_joints])
        valid_coords_fp16 = valid_coords.astype(np.float16)
        diff = np.nanmax(np.abs(valid_coords - valid_coords_fp16))
        if diff > 1:
            valid_coords_fp16 = valid_coords_fp16[:0]
        else:
            valid_coords = valid_coords[:0]

    if ex.mask is None:
        mask_zcounts = b''
    elif 'counts' in ex.mask:
        mask_zcounts = zlib.compress(ex.mask['counts'])
    elif 'zcounts' in ex.mask:
        mask_zcounts = ex.mask['zcounts']
    else:
        raise ValueError('No counts in mask')

    if ex.mask is not None:
        imshape = ex.mask['size']
    else:
        imshape = improc.image_extents(ex.image_path)[::-1]

    if i_valid_joints.shape[0] == ex.world_coords.shape[0]:
        i_valid_joints = i_valid_joints[:0]

    return dict(
        image_path=ex.image_path,
        imshape=imshape,
        mask_zipped_runlengths=np.frombuffer(mask_zcounts, dtype=np.uint8),
        bbox=np.round(ex.bbox),
        pose3d_world=dict(
            coords_fp32=valid_coords.ravel(),
            coords_fp16=valid_coords_fp16.ravel(),
            row_indices=i_valid_joints
        ),
        camera=dict(
            rotvec_world_to_camera=cv2.Rodrigues(ex.camera.R)[0][:, 0],
            optical_center=ex.camera.t,
            intrinsic_matrix=ex.camera.intrinsic_matrix[:2],
            distortion_coeffs=ex.camera.distortion_coeffs if ex.camera.distortion_coeffs is not None else np.zeros(
                (0,), dtype=np.float32)
        )
    )


if __name__ == '__main__':
    main()
