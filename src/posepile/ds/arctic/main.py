import argparse
import zipfile

import boxlib
import cameralib
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import smpl.numpy
import trimesh
from simplepyutils import FLAGS
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

DATASET_NAME = 'arctic'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_dataset_unsegmented()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache(f'{DATASET_NAME}_unsegmented.pkl', min_time="2023-12-01T13:35:22")
def make_dataset_unsegmented():
    betas_per_person, gender_per_person = get_betas_and_genders()
    cameras_per_person = get_cameras()
    splitinfo = spu.load_json(f'{DATASET_DIR}/splits_json/protocol_p1.json')
    seq_ids = splitinfo['train'] + splitinfo['val']

    examples_train = []
    examples_val = []
    with spu.ThrottledPool() as pool:
        for seq_id in spu.progressbar(seq_ids):
            person_id = seq_id.split('/')[0]
            gender = gender_per_person[person_id]
            bm = smpl.numpy.get_cached_body_model('smplx', gender)
            cameras = cameras_per_person[person_id]
            betas = betas_per_person[person_id]
            smplx_data = np.load(
                f'{DATASET_DIR}/raw_seqs/{seq_id}.smplx.npy', allow_pickle=True).item()
            poses = get_full_pose(smplx_data)
            jointss = bm(
                poses, np.repeat(betas[np.newaxis], len(poses), axis=0), smplx_data['transl'],
                return_vertices=False)['joints'] * 1000

            zippath = f'{DATASET_DIR}/images_zips/{seq_id}.zip'
            detections_in_zip = spu.load_pickle(zippath.replace('.zip', '.pkl'))
            with zipfile.ZipFile(zippath) as zipf:
                for i_cam, cam in enumerate(cameras):
                    sampler = AdaptivePoseSampler2(100, True, True, 100)
                    for i_frame, joints in enumerate(
                            spu.progressbar(jointss, desc=f'{seq_id}/{i_cam}')):
                        if sampler.should_skip(joints):
                            continue

                        parameters = dict(
                            type='smplx', gender=gender, pose=poses[i_frame], shape=betas,
                            trans=smplx_data['transl'][i_frame])
                        joints2d = cam.world_to_image(joints)
                        bbox_gt = boxlib.expand(boxlib.bb_of_points(joints2d), 1.1)
                        relpath_in_zip = f'{i_cam + 1}/{i_frame + 1:05d}.jpg'

                        try:
                            detections = detections_in_zip[relpath_in_zip][:, :4]
                        except:
                            print('No dets for', relpath_in_zip)
                            continue
                        if len(detections) == 0:
                            continue
                        im = np.frombuffer(zipf.open(relpath_in_zip).read(), np.uint8)
                        ious = [boxlib.iou(det, bbox_gt) for det in detections]
                        if np.max(ious) < 0.1:
                            continue
                        bbox = detections[np.argmax(ious)]

                        ex = ds3d.Pose3DExample(
                            image_path=im, camera=cam, parameters=parameters, world_coords=None,
                            bbox=bbox)
                        new_image_relpath = (
                            f'arctic_downscaled/{seq_id}/{i_cam + 1}/{i_frame + 1:05d}.jpg')
                        exs = examples_train if seq_id in splitinfo['train'] else examples_val
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_relpath),
                            callback=exs.append)

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples_train, examples_val)


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    ds = make_dataset_unsegmented()
    ds3d.add_masks(
        ds, f'{DATA_ROOT}/{DATASET_NAME}_downscaled/masks',
        relative_root=f'{DATA_ROOT}/{DATASET_NAME}_downscaled')
    return ds


def get_cameras():
    metadata = spu.load_json(f'{DATASET_DIR}/meta/misc.json')
    result = {}
    for i_subj in range(10):
        subj_id = f's{i_subj + 1:02d}'
        intrinsic_matrices = np.array(metadata[subj_id]['intris_mat'])
        extrinsic_matrices = np.array(metadata[subj_id]['world2cam'])
        result[subj_id] = [
            cameralib.Camera(intrinsic_matrix=intr, extrinsic_matrix=extr, world_up=(0, 0, 1)) for
            intr, extr in zip(intrinsic_matrices, extrinsic_matrices)]
        for c in result[subj_id]:
            c.t *= 1000
    return result


def get_full_pose(data):
    pose_parts = [
        'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
        'left_hand_pose', 'right_hand_pose']
    return np.concatenate([data[x].astype(np.float32) for x in pose_parts], axis=-1)


def compute_beta_vector(v_template_custom, gender, n_betas=16):
    bm = smpl.numpy.get_cached_body_model('smplx', gender)
    v_template_default = bm.single(shape_betas=np.zeros([0], np.float32))['vertices']
    delta = v_template_custom - v_template_default
    betas, res_err, _, _ = np.linalg.lstsq(
        bm.shapedirs[:, :, :n_betas].reshape(bm.num_vertices * 3, -1), delta.reshape(-1),
        rcond=None)
    return betas


@spu.picklecache(f'{DATASET_NAME}_betas.pkl', min_time="2023-12-01T13:35:22")
def get_betas_and_genders():
    metadata = spu.load_json(f'{DATASET_DIR}/meta/misc.json')
    betas = {}
    genders = {}
    for i in range(10):
        subject_id = f's{i + 1:02d}'
        genders[subject_id] = metadata[subject_id]['gender']
        v_template_custom = np.array(
            trimesh.load(f'{DATASET_DIR}/meta/subject_vtemplates/{subject_id}.obj').vertices,
            np.float32)
        betas[subject_id] = compute_beta_vector(v_template_custom, genders[subject_id], n_betas=16)
    return betas, genders


if __name__ == '__main__':
    main()
