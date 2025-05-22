import argparse
import functools
import glob
import io
import mmap
import multiprocessing
import os
import os.path as osp
import pickle
import subprocess
import warnings
import zipfile

import boxlib
import cameravision
import numpy as np
import posepile.datasets3d as ds3d
import posepile.joint_info
import posepile.util.improc as improc
import requests
import simplepyutils as spu
import smpl.numpy
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS, logger
from posepile.util.parallel_map import parallel_map_as_generator

DATASET_ROOT = f'{DATA_ROOT}/moyo'
USER_EMAIL = 'istvan.sarandi@uni-tuebingen.de'
USER_PASSWORD = ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache('moyo_stage1.pkl', min_time="2023-11-12T20:19:34")
def make_stage1():
    relpaths_train = load_relpaths(
        f'{DATASET_ROOT}/moyo_toolkit/moyo/bash/assets/urls/images_train.txt')
    relpaths_val = load_relpaths(
        f'{DATASET_ROOT}/moyo_toolkit/moyo/bash/assets/urls/images_val.txt')
    relpaths = relpaths_train + relpaths_val

    examples = []
    relpaths_not_done = []
    for relpath in relpaths:
        learning_phase = 'train' if 'images/train' in relpath else 'val'

        try:
            exs = spu.load_pickle(f'{DATA_ROOT}/cache/moyo/{learning_phase}/{relpath}.pkl')
        except FileNotFoundError as e:
            relpaths_not_done.append(relpath)
        else:
            examples.extend(exs)

    downloaded_files = parallel_map_as_generator((
        (download_file, (), dict(relpath=relpath, out_root='/fastwork/moyo'))
        for relpath in relpaths_not_done),
        n_workers=1, max_unconsumed=1)

    for relpath, abspath in zip(relpaths_not_done, downloaded_files):
        examples.extend(process_image_zip(relpath, abspath))

    examples_train = [ex for ex in examples if ex.image_path.startswith('moyo_downscaled/train')]
    examples_val = [ex for ex in examples if ex.image_path.startswith('moyo_downscaled/val')]

    return ds3d.Pose3DDataset(
        ds3d.JointInfo([], []),
        examples_train, examples_val)


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('moyo.pkl', min_time="2023-11-13T03:51:26")
def make_dataset():
    ds = make_stage1()
    ds3d.add_masks(
        ds, f'{DATA_ROOT}/moyo_downscaled/masks',
        relative_root=f'{DATA_ROOT}/moyo_downscaled')
    return ds


def process_image_zip(relpath, abspath):
    logger.info(f'Processing {relpath}')
    learning_phase = 'train' if 'images/train' in relpath else 'val'
    camera_maps = {}
    camera_maps['220923'] = load_cameras(glob.glob(
        f'{DATASET_ROOT}/MOYO/20220923_20220926_with_hands/'
        f'cameras/20220923/*/cameras_param.json')[0])
    camera_maps['220926'] = load_cameras(glob.glob(
        f'{DATASET_ROOT}/MOYO/20220923_20220926_with_hands/'
        f'cameras/20220926/*/cameras_param.json')[0])
    camera_maps['221004'] = load_cameras(glob.glob(
        f'{DATASET_ROOT}/MOYO/20221004_with_com/'
        f'cameras/20221004/*/cameras_param.json')[0])
    session_id_to_dirname = {
        '220923': '20220923_20220926_with_hands',
        '220926': '20220923_20220926_with_hands',
        '221004': '20221004_with_com'}

    bm = smpl.numpy.get_cached_body_model('smplx', 'female')
    moyoshape = get_beta_of_moyo_subject()

    examples = []

    pose_sampler = AdaptivePoseSampler2(
        0.1, True, True, 200)

    with spu.ThrottledPool() as pool:
        with zipfile.ZipFile(abspath) as f:
            for fp in spu.progressbar(sorted(f.filelist, key=lambda x: x.filename)):
                if not fp.filename.endswith('.jpg'):
                    continue

                img_name = osp.basename(fp.filename)
                sessionid = img_name.split('_')[0]
                seq_name = spu.split_path(fp.filename)[0]
                i_frame = int(img_name.split('_')[-1].split('.')[0])

                # Ignore first 50 frames (person still in T-pose)
                if i_frame < 50:
                    continue

                cameras = camera_maps[sessionid]
                cam_num = int(spu.split_path(fp.filename)[-2].split('_')[-1])
                cam_id = f'cam_{cam_num}'
                camera = cameras[cam_id]

                try:
                    smplx_data = load_pickle(
                        f'{DATASET_ROOT}/MOYO/{session_id_to_dirname[sessionid]}/'
                        f'mosh/{learning_phase}/{seq_name}_stageii.pkl')
                except FileNotFoundError:
                    logger.warning(f'Could not find {seq_name}_stageii.pkl')
                    continue
                except pickle.UnpicklingError:
                    logger.warning(f'Could not unpickle {seq_name}_stageii.pkl')
                    continue

                # Ignore last 50 frames (person again in T-pose)
                if len(smplx_data['fullpose']) <= (i_frame + 50) * 2:
                    continue

                joints = bm.single(
                    smplx_data['fullpose'][i_frame * 2], moyoshape,
                    smplx_data['trans'][i_frame * 2], return_vertices=False)['joints']
                cam_joints = camera.world_to_camera(joints)
                cam_joints -= cam_joints[:1]

                if pose_sampler.should_skip(cam_joints):
                    continue

                parameters = dict(
                    type='smplx', gender='female', pose=smplx_data['fullpose'][i_frame * 2],
                    shape=moyoshape, trans=smplx_data['trans'][i_frame * 2])

                joints2d = camera.world_to_image(joints)
                bbox = boxlib.expand(boxlib.bb_of_points(joints2d), 1.05)

                image = improc.decode_jpeg_bytes(f.read(fp))
                camera2 = camera.copy()
                camera2.t *= 1000
                ex = ds3d.Pose3DExample(
                    image_path=image, camera=camera2, bbox=bbox, parameters=parameters,
                    world_coords=None)
                new_image_relpath = f'moyo_downscaled/{learning_phase}/{fp.filename}'

                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath),
                    callback=examples.append)
        logger.info('Waiting for all examples to be processed')

    logger.info(f'Writing cache for {relpath}')
    spu.dump_pickle(examples, f'{DATA_ROOT}/cache/moyo/{learning_phase}/{relpath}.pkl')
    os.remove(abspath)
    return examples


def download_cameras_and_smplx(session):
    relpaths = [
        f'MOYO/{sess}/{kind}.zip'
        for sess in ['20220923_20220926_with_hands', '20221004_with_com']
        for kind in ['cameras', 'mosh']]

    for relpath in relpaths:
        extract_and_delete_zip(
            download_file(relpath=relpath, out_root=f'{DATASET_ROOT}', session=session))


def load_cameras(path):
    j = spu.load_json(path)

    def load_camera(d):
        f = d['focal']
        optical_center = np.array(d['position'], np.float32) / 1000
        principal_point = d['princpt']
        R = np.array(d['rotation'], np.float32)
        intrinsic_matrix = np.array([
            [f, 0, principal_point[0]],
            [0, f, principal_point[1]],
            [0, 0, 1]
        ], np.float32)
        c = cameravision.Camera(
            intrinsic_matrix=intrinsic_matrix,
            optical_center=np.array(optical_center, np.float32),
            rot_world_to_cam=R)
        c.scale_output(0.5)
        return c

    return {k: load_camera(v) for k, v in j.items()}


def load_relpaths(url_path):
    urls = spu.read_lines(url_path)
    urls = [url.strip() for url in urls if url.strip()]
    relpaths = [url.split('=')[-1] for url in urls]
    return relpaths


def create_custom_smplx():
    """MOYO is a single-person dataset and the template of SMPLX is replaced with a custom one,
    instead of parameterizing her shape with betas.
    """
    import trimesh
    smplx_array = np.load(
        f'{DATA_ROOT}/body_models/smplx/SMPLX_FEMALE.npz', allow_pickle=True)
    smplx_array = dict(smplx_array)
    mesh = trimesh.load(
        osp.join(moyo_root,
                 'moyo_toolkit/data/v_templates/220923_yogi_03596_minimal_simple_female/mesh.ply'))
    custom_v_template = mesh.vertices
    smplx_array['v_template'] = custom_v_template
    np.savez(f'{DATA_ROOT}/body_models/smplxmoyo/SMPLX_FEMALE.npz', **smplx_array)


def download_file(relpath, out_root):
    out_path = osp.join(out_root, relpath)
    if is_zipfile_ok(out_path):
        logger.info(f'Already downloaded {relpath}')
        return out_path

    url = f'https://download.is.tue.mpg.de/download.php?domain=moyo&resume=1&sfile={relpath}'
    response = requests.post(
        url, data=dict(username=USER_EMAIL, password=USER_PASSWORD), stream=True, verify=False,
        allow_redirects=True)
    if response.status_code == 401:
        raise PermissionError(
            'Authentication failed. Please check your username and password.')

    total_length = int(response.headers.get('content-length'))
    logger.info(f'Downloading {relpath}')

    spu.ensure_parent_dir_exists(out_path)
    with open(out_path, 'wb') as f:
        for chunk in spu.progressbar(
                response.iter_content(chunk_size=8192), total=total_length // 8192):
            f.write(chunk)
    logger.info(f'Downloaded {relpath}')
    return out_path


def extract_and_delete_zip(abspath):
    try:
        with zipfile.ZipFile(abspath) as f:
            f.extractall(osp.dirname(abspath))
    except zipfile.BadZipFile as e:
        subprocess.check_call(['unzip', '-o', abspath, '-d', osp.dirname(abspath)])

    os.remove(abspath)


def is_zipfile_ok(abspath):
    try:
        with zipfile.ZipFile(abspath) as f:
            return f.testzip() is None
    except zipfile.BadZipfile:
        return False
    except FileNotFoundError:
        return False


@functools.lru_cache()
def load_pickle(file_path):
    return spu.load_pickle(file_path)


def compute_beta_vector(v_template_custom, gender, n_betas=16):
    bm = smpl.numpy.get_cached_body_model('smplx', gender)
    v_template_default = bm.single(shape_betas=np.zeros([0], np.float32))['vertices']
    delta = v_template_custom - v_template_default
    A = bm.shapedirs[:, :, :n_betas].reshape(bm.num_vertices * 3, -1)
    b = delta.reshape(-1)
    betas, res_err, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return betas


@functools.lru_cache()
def get_beta_of_moyo_subject():
    custom_v_template = trimesh.load(
        f'{DATA_ROOT}/moyo/moyo_toolkit/data/v_templates/'
        f'220923_yogi_03596_minimal_simple_female/mesh.ply').vertices
    return compute_beta_vector(custom_v_template, 'female', n_betas=128)


if __name__ == '__main__':
    main()
