import os.path as osp
import more_itertools
import boxlib
import h5py
import numpy as np
import simplepyutils as spu
import smpl.tensorflow.full_fitting
import tensorflow as tf

import posepile.datasets3d as ds3d
from posepile.paths import DATA_ROOT

DATASET_NAME = 'dfaust'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'
RENDER_DIR = f'{DATA_ROOT}/{DATASET_NAME}_render'


def main():
    make_dataset()


@spu.picklecache(f'{DATASET_NAME}_render.pkl', min_time="2023-12-21T10:50:29")
def make_dataset():
    examples = []
    camera_paths = spu.sorted_recursive_glob(f'{RENDER_DIR}/*/cameras.pkl')

    registrations = dict(
        male=h5py.File(f'{DATASET_DIR}/registrations_m.hdf5', 'r'),
        female=h5py.File(f'{DATASET_DIR}/registrations_f.hdf5', 'r'))

    J_regressor_post_lbs_m = np.load(f'/work/sarandi/smpl_male_J_regressor_post_lbs.npy')
    J_regressor_post_lbs_f = np.load(f'/work/sarandi/smpl_female_J_regressor_post_lbs.npy')

    bms = dict(
        male=smpl.tensorflow.get_cached_body_model('smpl', 'male'),
        female=smpl.tensorflow.get_cached_body_model('smpl', 'female'))

    fitters = dict(
        male=smpl.tensorflow.full_fitting.Fitter(bms['male'], J_regressor_post_lbs_m,
                                                 num_betas=128),
        female=smpl.tensorflow.full_fitting.Fitter(bms['female'], J_regressor_post_lbs_f,
                                                   num_betas=128))

    input_signature = [
        tf.TensorSpec(shape=(None, bms['male'].num_vertices, 3), dtype=tf.float32, name='to_fit')]

    @tf.function(input_signature=input_signature)
    def fit_m(verts):
        return fitters['male'].fit(verts, n_iter=4, l2_regularizer=5e-6)

    @tf.function(input_signature=input_signature)
    def fit_f(verts):
        return fitters['female'].fit(verts, n_iter=4, l2_regularizer=5e-6)

    fit_fns = dict(male=fit_m, female=fit_f)

    for camera_path in spu.progressbar(camera_paths):
        seq_id = osp.basename(osp.dirname(camera_path))
        g = 'male' if seq_id in registrations['male'].keys() else 'female'
        verts_seq = registrations[g][seq_id][:].transpose(2, 0, 1)

        poses = []
        shapes = []
        transs = []
        for verts_batch in more_itertools.chunked(verts_seq, 32):
            fit_result = fit_fns[g](verts_batch)
            poses.append(fit_result['pose_rotvecs'].numpy())
            shapes.append(fit_result['shape_betas'].numpy())
            transs.append(fit_result['trans'].numpy())

        pose = np.concatenate(poses, axis=0)
        shape = np.concatenate(shapes, axis=0)
        trans = np.concatenate(transs, axis=0)

        reconstr = bms[g](pose, shape, trans)['vertices'].numpy()
        errors = np.linalg.norm(reconstr - verts_seq, axis=-1)
        print(f'{seq_id}: {np.mean(errors):.4f} +- {np.std(errors):.4f}, max {np.max(errors):.4f}')

        cameras = spu.load_pickle(camera_path)
        masks = spu.load_pickle(f'{RENDER_DIR}/{seq_id}/masks.pkl')

        frame_paths = spu.sorted_recursive_glob(f'{RENDER_DIR}/{seq_id}/*.jpg')
        i_gen_to_i_frame = {}
        for frame_path in frame_paths:
            base_noext = osp.splitext(osp.basename(frame_path))[0]
            parts = base_noext.split('_')
            i_gen = int(parts[0])
            i_frame = int(parts[1])
            i_gen_to_i_frame[i_gen] = i_frame

        for i_gen, (camera, mask) in enumerate(zip(cameras, masks)):
            i_frame = i_gen_to_i_frame[i_gen]
            impath = f'{RENDER_DIR}/{seq_id}/{i_gen:06d}_{i_frame:06d}.jpg'
            image_relpath = osp.relpath(impath, RENDER_DIR)
            bbox = boxlib.bb_of_mask(mask)

            parameters = dict(
                type='smpl', gender=g, pose=pose[i_frame], shape=shape[i_frame],
                trans=trans[i_frame])
            ex = ds3d.Pose3DExample(
                image_path=f'{DATASET_NAME}_render/{image_relpath}',
                camera=camera, bbox=bbox,
                parameters=parameters, world_coords=None, mask=mask)
            examples.append(ex)

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples)


if __name__ == '__main__':
    main()
