import argparse
import glob
import os.path as osp
import random

import boxlib
import cameralib
import numpy as np
import posepile.datasets3d as ds3d
import posepile.joint_info
import scipy.spatial.distance
import simplepyutils as spu
import smpl.numpy
from posepile.ds.dna_rendering.smc.SMCReader import SMCReader
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS

DATASET_NAME = 'dna_rendering'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    while True:
        main_paths = sorted(glob.glob(f'{DATASET_DIR}/dna_rendering_part*_main/*.smc'))
        random.shuffle(main_paths)
        all_done = all(osp.exists(get_out_path(main_path)) for main_path in main_paths)
        if all_done:
            return
        for main_path in main_paths:
            dirname = osp.basename(osp.dirname(main_path))
            seq_id = osp.splitext(osp.basename(main_path))[0]
            part = 1 if 'part1' in dirname else 2
            out_id = f'{part}_{seq_id}'
            if not osp.exists(f'{DATA_ROOT}/dna_rendering_downscaled/{out_id}'):
                try:
                    process_main(main_path)
                except:
                    print(f'Error processing {main_path}')


def process_main(main_path):
    selector = [1, 2, 4, 5, 7, 8, *range(16, 22), *range(25, 144)]

    dirname = osp.basename(osp.dirname(main_path))
    seq_id = osp.splitext(osp.basename(main_path))[0]
    out_id = get_out_id(main_path)
    out_path = get_out_path(main_path)
    if osp.exists(out_path):
        return
    examples = []

    smc_main = SMCReader(main_path)
    smc_anno = SMCReader(
        f'{DATASET_DIR}/{dirname.replace("main", "annotations")}/{seq_id}_annots.smc')

    n_frames = smc_main.get_Camera_12mp_info()['num_frame']
    smplx_bm = smpl.numpy.get_cached_body_model(
        'smplx', smc_main.get_actor_info()['gender'])

    with spu.ThrottledPool() as pool:
        for cam_id in spu.progressbar(range(60), desc=f'Cameras of {out_id}'):
            cam_type = 'Camera_5mp' if cam_id < 48 else 'Camera_12mp'
            pose_sampler = AdaptivePoseSampler2(
                100, True, True, 200)

            calib = smc_anno.get_Calibration(cam_id)
            camera = cameralib.Camera(
                intrinsic_matrix=calib['K'],
                rot_world_to_cam=calib['RT'][:3, :3].T,
                optical_center=calib['RT'][:3, 3] * 1000,
                distortion_coeffs=calib['D'], world_up=(0, -1, 0))

            for i_frame in spu.progressbar(range(n_frames), desc=f'cam {cam_id}', leave=False):
                smplx_params = smc_anno.get_SMPLx(i_frame)
                smplx_joints = smplx_bm.single(
                    smplx_params['fullpose'], smplx_params['betas'], smplx_params['transl'],
                    return_vertices=False)['joints'] * 1000
                if pose_sampler.should_skip(camera.world_to_camera(smplx_joints)):
                    continue

                kp3d = smc_anno.get_Keypoints3d(i_frame)
                kp3d = replace_nearby_points_by_nan(kp3d, tolerance=5e-4)
                kp3d = kp3d[selector, :3] * 1000

                parameters = dict(
                    type='smplx', gender=smc_main.get_actor_info()['gender'],
                    pose=smplx_params['fullpose'].reshape(-1), shape=smplx_params['betas'],
                    expression=smplx_params['expression'],
                    # the scale should not be used, else the result is not aligned with the images
                    # scale=np.float32(smplx_params['scale']),
                    trans=smplx_params['transl'])
                mask = smc_anno.get_mask(cam_id, i_frame)
                box = boxlib.bb_of_mask(mask > 127)

                im_bytes = smc_main.smc[cam_type][str(cam_id)]['color'][str(i_frame)][:]
                # im = decode_jpeg(im_bytes)
                # drawing.draw_box(im, box, color=(0, 255, 0))
                # pose2d = camera.world_to_image(kp3d)
                # for i_joint, (x, y) in enumerate(pose2d):
                #    drawing.circle(im, (x, y), radius=5, color=(255, 0, 0))

                ex = ds3d.Pose3DExample(
                    image_path=im_bytes, camera=camera, bbox=box,
                    world_coords=kp3d[:, :3], mask=mask, parameters=parameters)
                new_image_relpath = (
                    f'dna_rendering_downscaled/{out_id}/{cam_id:02d}/{i_frame:06d}.jpg')
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath),
                    kwargs=dict(
                        downscale_input_for_antialias=True, downscale_at_decode=False,
                        reuse_image_array=True),
                    callback=examples.append)

    spu.dump_pickle(examples, out_path)


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-11-25T13:25:13")
def make_dataset():
    examples = []
    for p in spu.progressbar(
            spu.sorted_recursive_glob(f'{DATA_ROOT}/dna_rendering_downscaled/stage1/*.pkl')):
        examples.extend(spu.load_pickle(p))

    ds = ds3d.Pose3DDataset(get_joint_info(), examples)
    ds3d.filter_dataset_by_plausibility(ds)
    return ds


def get_joint_info():
    return posepile.joint_info.JointInfo(
        'lhip,rhip,lkne,rkne,lank,rank,lsho,rsho,lelb,relb,lwri,rwri,lhan5,lhan6,lhan7,lhan9,'
        'lhan10,lhan11,lhan17,lhan18,lhan19,lhan13,lhan14,lhan15,lhan1,lhan2,lhan3,rhan5,rhan6,'
        'rhan7,rhan9,rhan10,rhan11,rhan17,rhan18,rhan19,rhan13,rhan14,rhan15,rhan1,rhan2,rhan3,'
        'nose,reye,leye,rear,lear,ltoe,ltoe2,lhee,rtoe,rtoe2,rhee,lhan4,lhan8,lhan12,lhan16,'
        'lhan20,rhan4,rhan8,rhan12,rhan16,rhan20,rface9,rface10,rface11,rface12,rface13,lface13,'
        'lface12,lface11,lface10,lface9,nose1,nose2,nose3,nose4,rface14,rface15,nose5,lface15,'
        'lface14,reye1,reye2,reye3,reye4,reye5,reye6,leye4,leye3,leye2,leye1,leye6,leye5,rmouth1,'
        'rmouth2,rmouth3,mouth1,lmouth3,lmouth2,lmouth1,lmouth4,lmouth5,mouth2,rmouth5,rmouth4,'
        'rmouth6,rmouth7,mouth3,lmouth7,lmouth6,lmouth8,mouth4,rmouth8,rface1,rface2,rface3,'
        'rface4,rface5,rface6,rface7,rface8,chin,lface8,lface7,lface6,lface5,lface4,lface3,'
        'lface2,lface1',
        'rhip-rkne-rank-rhee,'
        'rank-rtoe,'
        'rank-rtoe2,'
        'nose-reye-rear,'
        'rsho-relb-rwri-rhan1-rhan2-rhan3-rhan4,'
        'rwri-rhan5-rhan6-rhan7-rhan8,'
        'rwri-rhan9-rhan10-rhan11-rhan12,'
        'rwri-rhan13-rhan14-rhan15-rhan16,'
        'rwri-rhan17-rhan18-rhan19-rhan20,'
        'nose1-nose2-nose3-nose4-nose5,'
        'rface14-rface15-nose5,'
        'reye1-reye2-reye3-reye4-reye5-reye6-reye1,'
        'mouth2-rmouth5-rmouth4-rmouth1-rmouth2-rmouth3-mouth1,'
        'mouth4-rmouth8-rmouth6-rmouth7-mouth3,'
        'rface1-rface2-rface3-rface4-rface5-rface6-rface7-rface8-chin,'
        'rface9-rface10-rface11-rface12-rface13')


def get_out_id(main_path):
    dirname = osp.basename(osp.dirname(main_path))
    seq_id = osp.splitext(osp.basename(main_path))[0]
    part = 1 if 'part1' in dirname else 2
    out_id = f'{part}_{seq_id}'
    return out_id


def get_out_path(main_path):
    return f'{DATA_ROOT}/dna_rendering_downscaled/stage1/{get_out_id(main_path)}.pkl'


def replace_nearby_points_by_nan(points, tolerance=5e-4):
    points = points.copy()
    distances = scipy.spatial.distance.cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    close_points_indices = np.argwhere(np.any(distances < tolerance, axis=1))[:, 0]
    points[close_points_indices] = np.nan
    return points


if __name__ == '__main__':
    main()
