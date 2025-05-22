import functools

import barecat
import more_itertools
import numpy as np
import simplepyutils as spu
import smpl
import smpl.tensorflow.full_fitting
import tensorflow as tf
from posepile.paths import DATA_ROOT


def main():
    smplx_neutral = smpl.tensorflow.get_cached_body_model('smplx', 'neutral')
    smplx_call = tf.function(smplx_neutral.__call__, input_signature=[
        tf.TensorSpec(
            shape=(None, 3 * smplx_neutral.num_joints), dtype=tf.float32, name='pose_rotvecs'),
        tf.TensorSpec(
            shape=(None, None), dtype=tf.float32, name='shape_betas'),
        tf.TensorSpec(
            shape=(None, 3), dtype=tf.float32, name='trans'),
        tf.TensorSpec(
            shape=(None,), dtype=tf.float32, name='kid_factor'),
    ])

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=(None, None), dtype=tf.float32, name='shape_betas')])
    def smplx_t_pose(shape_betas):
        return smplx_neutral(shape_betas=shape_betas)

    input_signature = [
        tf.TensorSpec(shape=(None, smplx_neutral.num_vertices, 3),
                      dtype=tf.float32, name='to_fit'),
        tf.TensorSpec(shape=(), dtype=tf.int32, name='n_iter'),
        tf.TensorSpec(shape=(), dtype=tf.float32, name='l2_regularizer'),
        tf.TensorSpec(shape=(None, smplx_neutral.num_vertices, 3),
                      dtype=tf.float32, name='initial_latent'),
    ]
    projdir = f'{DATA_ROOT}/projects/localizerfields'
    J_regressor_post_lbs = np.load(f'{projdir}/smplx_neutral_J_regressor_post_lbs.npy')

    smplx_neutral_adult_fitter = tf.function(smpl.tensorflow.full_fitting.Fitter(
        smplx_neutral, J_regressor_post_lbs, num_betas=64,
        enable_kid=False).fit, input_signature=input_signature)
    smplx_neutral_kid_fitter = tf.function(smpl.tensorflow.full_fitting.Fitter(
        smplx_neutral, J_regressor_post_lbs, num_betas=64,
        enable_kid=True).fit, input_signature=input_signature)

    virtual_markers = np.load(f'{projdir}/smplx_256_acae_nofit.npz')

    # smplx_neutral_adult_fitter = tf.function(smpl.tensorflow.fitting.Fitter(
    #     smplx_neutral, virtual_markers['w1'], virtual_markers['w2'], J_regressor_post_lbs,
    #     num_betas=300, enable_kid=False).fit_latent)
    #
    # smplx_neutral_kid_fitter = tf.function(smpl.tensorflow.fitting.Fitter(
    #     smplx_neutral, virtual_markers['w1'], virtual_markers['w2'], J_regressor_post_lbs,
    #     num_betas=300, enable_kid=True).fit_latent)

    smpl2smplx_mat = (spu.load_pickle(
        f'{DATA_ROOT}/bedlam/smplx/transfer_data/smpl2smplx_deftrafo_setup.pkl'
    )['mtx'].tocsr()[:, :6890]).astype(np.float32)
    enc_smpl = virtual_markers['w1'].T @ smpl2smplx_mat
    enc_smplx = virtual_markers['w1']

    smpl2smplx_mat_coo = smpl2smplx_mat.tocoo()
    m_o2x_tf = tf.SparseTensor(
        indices=np.stack([smpl2smplx_mat_coo.row, smpl2smplx_mat_coo.col], axis=-1),
        values=smpl2smplx_mat_coo.data, dense_shape=smpl2smplx_mat_coo.shape)

    with (barecat.Barecat(
            f'{DATA_ROOT}/bc/merged_smplx_converted3.barecat', auto_codec=True,
            readonly=False, overwrite=True) as writer,
        barecat.Barecat(
            f'{DATA_ROOT}/bc/merged_parametric.barecat', auto_codec=True) as reader):

        pbar = spu.progressbar_items(reader)
        for batch in more_itertools.chunked(pbar, 128):
            batch = dict(batch)
            for path in list(batch):
                if False and path in writer:
                    batch.pop(path)
                    continue

                data = batch[path]
                if ('parameters' not in data or
                        (data['parameters']['type'] == 'smplx' and
                         data['parameters']['gender'] == 'neutral')):
                    writer[path] = data
                    batch.pop(path)

            if not batch:
                continue

            # Group the batch according to what body model (kind and gender) we need
            # to use to obtain the vertices
            bm_to_paths = spu.groupby(
                batch.keys(), lambda x: (
                    batch[x]['parameters']['type'],
                    batch[x]['parameters']['gender']))

            # Obtain the vertices for each group
            path_to_target_verts = {}
            path_to_initial_latents = {}
            for (model_name, gender), paths in bm_to_paths.items():
                subbatch = {path: batch[path] for path in paths}
                if not subbatch:
                    continue

                pose, shape = (
                    np.stack([batch[path]['parameters'][name] for path in paths])
                    for name in ['pose', 'shape'])
                trans = np.stack([
                    batch[path]['parameters']['trans'] / (
                        1000 if batch[path]['impath'].startswith('surreal') else 1)
                    for path in paths])
                kid_factor = np.stack([
                    batch[path]['parameters']['kid_factor']
                    if 'kid_factor' in batch[path]['parameters']
                    else np.float32(0)
                    for path in paths])

                # Get the original vertices
                body_model = get_cached_body_model_forward(model_name, gender)
                verts_target = body_model(pose, shape, trans, kid_factor)['vertices']

                converter = (
                    m_o2x_tf if model_name != 'smplx' else
                    tf.sparse.eye(smplx_neutral.num_vertices))
                C_o2x, t_o2x = get_o2x(model_name, gender, converter)
                initial_betas_smplx = tf.einsum(
                    'Ss,bs->bS', C_o2x[:, :shape.shape[1]], shape) + t_o2x
                initial_vertices = smplx_t_pose(shape_betas=initial_betas_smplx)['vertices']
                initial_latent = initial_vertices

                verts_target = sparse_dense_matmul_batch(converter, verts_target)

                path_to_target_verts.update(dict(zip(paths, verts_target)))
                path_to_initial_latents.update(dict(zip(paths, initial_latent)))

            # Fit the SMPL-X neutral model to the obtained vertices,
            # separately for adults and kids
            for is_kid in [True, False]:
                subbatch = {
                    path: data for path, data in batch.items()
                    if (data['parameters'].get('kid_factor', 0) > 0) == is_kid}
                if not subbatch:
                    continue

                subbatch_target_verts = tf.stack(
                    [path_to_target_verts[path] for path in subbatch])
                subbatch_initial_latents = tf.stack(
                    [path_to_initial_latents[path] for path in subbatch])

                fitter = smplx_neutral_kid_fitter if is_kid else smplx_neutral_adult_fitter
                fit_result = fitter(
                    subbatch_target_verts, n_iter=4, l2_regularizer=5e-6,
                    initial_vertices=subbatch_initial_latents)

                kid_factors = fit_result['kid_factor'] if is_kid else tf.zeros_like(
                    fit_result['kid_factor'])
                converted_res = smplx_call(
                    fit_result['pose_rotvecs'], fit_result['shape_betas'], fit_result['trans'],
                    kid_factors)
                error = tf.norm(converted_res['vertices'] - subbatch_target_verts, axis=-1)
                mean_error = tf.reduce_mean(error)
                max_error = tf.reduce_max(error)
                perc90 = np.percentile(error.numpy(), 90)
                perc95 = np.percentile(error.numpy(), 95)
                pck = tf.reduce_mean(
                    tf.cast(error < 0.01, tf.float32))
                pck5 = tf.reduce_mean(
                    tf.cast(error < 0.005, tf.float32))
                ds_name = list(subbatch.keys())[0].split('/')[1]

                pbar.set_description(
                    f'Error: {mean_error * 1000:04.1f} (max {max_error * 1000:04.1f}, perc90 '
                    f'{perc90 * 1000:04.1f}, perc95 {perc95 * 1000:04.1f}), pck10: {pck:07.2%}, '
                    f'pck5: {pck5:07.2%} {ds_name}')

                for (path, data), pose, shape, trans, kid_factor in zip(
                        subbatch.items(),
                        fit_result['pose_rotvecs'].numpy(),
                        fit_result['shape_betas'].numpy(),
                        fit_result['trans'].numpy(),
                        fit_result['kid_factor'].numpy()):
                    data['parameters'].update(
                        dict(pose=pose, shape=shape, trans=trans))
                    if is_kid:
                        data['parameters']['kid_factor'] = kid_factor
                    data['parameters']['gender'] = 'neutral'
                    data['parameters']['type'] = 'smplx'
                    # writer[path] = data


@functools.lru_cache()
def get_cached_body_model_forward(model_name, gender):
    bm = smpl.tensorflow.get_cached_body_model(model_name, gender)
    input_signature = [
        tf.TensorSpec(
            shape=(None, 3 * bm.num_joints), dtype=tf.float32, name='pose_rotvecs'),
        tf.TensorSpec(
            shape=(None, None), dtype=tf.float32, name='shape_betas'),
        tf.TensorSpec(
            shape=(None, 3), dtype=tf.float32, name='trans'),
        tf.TensorSpec(
            shape=(None,), dtype=tf.float32, name='kid_factor'),
    ]
    return tf.function(bm.__call__, input_signature=input_signature)


@functools.lru_cache()
def get_o2x(model_name, gender, m_o2x, l2_regularizer_t=1e-2, l2_regularizer_c=1e-2):
    bm_o = smpl.tensorflow.get_cached_body_model(model_name, gender)
    bm_x = smpl.tensorflow.get_cached_body_model('smplx', 'neutral')
    x_template = bm_x(shape_betas=tf.zeros([1, 0]))['vertices'][0]
    o_template = bm_o(shape_betas=tf.zeros([1, 0]))['vertices'][0]

    S_x = bm_x.shapedirs
    S_o = bm_o.shapedirs

    t_o2x = tf.linalg.lstsq(
        tf.reshape(S_x, [bm_x.num_vertices * 3, -1]),
        tf.reshape(tf.sparse.sparse_dense_matmul(m_o2x, o_template) - x_template, [-1, 1]),
        l2_regularizer=l2_regularizer_t)[:, 0]

    m_o2x_S_o = tf.sparse.sparse_dense_matmul(m_o2x, tf.reshape(S_o, [bm_o.num_vertices, -1]))
    C_o2x = tf.linalg.lstsq(
        tf.reshape(S_x, [bm_x.num_vertices * 3, -1]),
        tf.reshape(m_o2x_S_o, [bm_x.num_vertices * 3, -1]), l2_regularizer=l2_regularizer_c)

    return C_o2x, t_o2x


@tf.function(input_signature=[
    tf.SparseTensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)])
def sparse_dense_matmul_batch(sparse, dense):
    return tf.map_fn(lambda x: tf.sparse.sparse_dense_matmul(sparse, x), dense,
                     parallel_iterations=1024)


if __name__ == '__main__':
    main()
