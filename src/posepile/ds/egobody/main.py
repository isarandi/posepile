import os.path as osp
import re

import boxlib
import cameralib
import numpy as np
import posepile.datasets3d as ds3d
import scipy.optimize
import simplepyutils as spu
import smpl.numpy
from posepile.ds.rich.add_parametric import load_smplx_params
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.improc import decode_jpeg_bytes
from posepile.util.preproc_for_efficiency import make_efficient_example
import zipfile
from posepile.util import drawing
import itertools
import csv
import pandas as pd
from posepile.datasets3d import TRAIN, VALID, TEST

DATASET_NAME = 'egobody'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    make_dataset()


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-12T11:42:10")
def make_dataset():
    examples = []
    detections_all = spu.load_pickle(f'{DATASET_DIR}/yolov4_detections.pkl')

    kinect_master_intr = spu.load_json(f'{DATASET_DIR}/kinect_cam_params/kinect_master/Color.json')
    kinect_subs_intr = [spu.load_json(
        f'{DATASET_DIR}/kinect_cam_params/kinect_sub_{i + 1}/Color.json')
        for i in range(4)]

    zipf_kinect = zipfile.ZipFile(f'{DATASET_DIR}/kinect_color.zip')
    zipf_ego = zipfile.ZipFile(f'{DATASET_DIR}/egocentric_color.zip')
    csv_rows = list(csv.reader(open(f'{DATASET_DIR}/data_info_release.csv')))

    data_split_info = pd.read_csv(f'{DATASET_DIR}/data_splits.csv')
    train_rec_names = set(data_split_info['train'])
    val_rec_names = set(data_split_info['val'])
    test_rec_names = set(data_split_info['test'])
    examples = {TRAIN: [], VALID: [], TEST: []}

    with spu.ThrottledPool() as pool:
        for (scene_name, gender0, gender1, start_frame, end_frame, gender_fpv,
             rec_name) in spu.progressbar(csv_rows[1:]):
            gender0 = gender0.split()[1]
            gender1 = gender1.split()[1]

            extrinsics_dir = f'{DATASET_DIR}/calibrations/{rec_name}/cal_trans'
            kinect_cams, world_to_holo_world = get_kinect_calibs(
                extrinsics_dir, kinect_master_intr, kinect_subs_intr, scene_name)

            frame_id_to_holocam, frame_id_to_holoimpath = get_hololens_metadata(
                rec_name, zipf_ego, world_to_holo_world)

            cam_names = ['master'] + [f'sub_{i + 1}' for i in range(4)] + ['hololens']
            pose_samplers = {
                n: AdaptivePoseSampler2(
                    0.1, True, True, 200)
                for n in cam_names}
            pose_samplers['hololens'] = AdaptivePoseSampler2(
                0.1, True, True, 200)
            for i_frame in spu.progressbar(
                    range(int(start_frame), int(end_frame) + 1), leave=False, desc=rec_name):
                both_smplx = get_both_smplx(rec_name, i_frame, gender0, gender1, kinect_cams[0])

                joints_world = []
                for smplx_params in both_smplx:
                    bm = smpl.numpy.get_cached_body_model('smplx', smplx_params['gender'])
                    joints_world.append(bm.single(
                        smplx_params['pose'], smplx_params['shape'], smplx_params['trans'],
                        return_vertices=False)['joints'])

                cams = [*kinect_cams]
                if i_frame in frame_id_to_holocam:
                    cams.append(frame_id_to_holocam[i_frame])

                for cam_name, cam in zip(cam_names, cams):
                    if cam is None:
                        continue

                    if cam_name == 'hololens':
                        impath = frame_id_to_holoimpath[i_frame]
                    else:
                        impath = (f'kinect_color/{rec_name}/'
                                  f'{cam_name}/frame_{i_frame:05d}.jpg')

                    try:
                        detections = detections_all[impath]
                    except KeyError:
                        print(f'No detection list found for {impath}')
                        continue

                    if len(detections) == 0:
                        continue

                    joints_cam = [cam.world_to_camera(j) for j in joints_world]
                    joints_im = [cam.world_to_image(j) for j in joints_world]
                    gt_boxes = np.array([boxlib.expand(boxlib.bb_of_points(j), 1.05)
                                         for j in joints_im])
                    iou_matrix = np.array([[boxlib.iou(gt_box, det[:4])
                                            for det in detections]
                                           for gt_box in gt_boxes])
                    gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
                    im = None

                    for i_person, i_det in zip(gt_indices, det_indices):
                        if iou_matrix[i_person, i_det] < 0.2:
                            continue
                        is_in_det = boxlib.contains(detections[i_det], joints_im[i_person])
                        if np.sum(is_in_det) < 6:
                            continue
                        box = detections[i_det][:4]

                        if pose_samplers[cam_name].should_skip(joints_cam[i_person]):
                            continue

                        if im is None:
                            zipf = zipf_ego if cam_name == 'hololens' else zipf_kinect
                            im = np.frombuffer(zipf.open(impath).read(), np.uint8)

                        # drawing.draw_box(im, detections[i_det], color=(0, 255, 0))
                        # for i_joint, (x, y) in enumerate(joints_im[i_person]):
                        #     color = (0, 0, 255) if is_in_det[i_joint] else (255, 0, 0)
                        #     drawing.circle(im, (x, y), radius=5, color=color)

                        ex = ds3d.Pose3DExample(
                            world_coords=None,
                            image_path=im, camera=cam, bbox=box, parameters=both_smplx[i_person])
                        relpath_noext = osp.splitext(impath)[0]
                        new_image_relpath = f'egobody_downscaled/{relpath_noext}_{i_person:02d}.jpg'
                        phase = (
                            TRAIN if rec_name in train_rec_names else
                            VALID if rec_name in val_rec_names else
                            TEST)
                        ex.camera = ex.camera.copy()
                        ex.camera.t *= 1000
                        pool.apply_async(
                            make_efficient_example, (ex, new_image_relpath),
                            #kwargs=dict(extreme_perspective=False),
                            callback=examples[phase].append)

    return ds3d.Pose3DDataset(
        ds3d.JointInfo([], []), examples[TRAIN], examples[VALID], examples[TEST])


def get_kinect_calibs(extrinsics_dir, kinect_master_intr, kinect_subs_intr, scene_name):
    world_to_kinect_master = np.linalg.inv(
        spu.load_json(f'{extrinsics_dir}/kinect12_to_world/{scene_name}.json')['trans'])
    kinect_master = cameralib.Camera(
        extrinsic_matrix=world_to_kinect_master,
        intrinsic_matrix=kinect_master_intr['camera_mtx'],
        distortion_coeffs=kinect_master_intr['k'], world_up=(0, 1, 0))

    kinect_subs = []
    for i, cam_num in enumerate([11, 13, 14, 15]):
        try:
            extr = np.linalg.inv(
                spu.load_json(f'{extrinsics_dir}/kinect_{cam_num}to12_color.json')['trans'])
        except FileNotFoundError:
            kinect_subs.append(None)
        else:
            cam = cameralib.Camera(
                extrinsic_matrix=extr @ world_to_kinect_master,
                intrinsic_matrix=kinect_subs_intr[i]['camera_mtx'],
                distortion_coeffs=kinect_subs_intr[i]['k'], world_up=(0, 1, 0))
            kinect_subs.append(cam)

    kinect_master_to_holo = np.linalg.inv(
        spu.load_json(f'{extrinsics_dir}/holo_to_kinect12.json')['trans'])
    world_to_holo_world = kinect_master_to_holo @ world_to_kinect_master
    return [kinect_master] + kinect_subs, world_to_holo_world


def get_both_smplx(recording_name, i_frame, gender0, gender1, ref_cam):
    pose, betas, trans, kid_factor, expression = load_smplx_params(
        f'{DATASET_DIR}/smplx/{recording_name}/body_idx_0/'
        f'results/frame_{i_frame:05d}/000.pkl')
    bm = smpl.numpy.get_cached_body_model('smplx', gender0)
    pose, trans = bm.rototranslate(ref_cam.R.T, ref_cam.t, pose, betas, trans)
    person0_params = dict(
        type='smplx', gender=gender0, pose=pose, shape=betas,
        expression=expression, kid_factor=np.float32(0), trans=trans)

    pose, betas, trans, kid_factor, expression = load_smplx_params(
        f'{DATASET_DIR}/smplx/{recording_name}/body_idx_1/'
        f'results/frame_{i_frame:05d}/000.pkl')
    pose, trans = bm.rototranslate(ref_cam.R.T, ref_cam.t, pose, betas, trans)
    person1_params = dict(
        type='smplx', gender=gender1, pose=pose, shape=betas,
        expression=expression, kid_factor=np.float32(0), trans=trans)
    return person0_params, person1_params


def get_hololens_metadata(recording_name, zipf_ego, world_to_holo_world):
    namelist = zipf_ego.namelist()
    datetime = osp.basename(
        next(iter(x for x in namelist if
                  x.startswith(f'egocentric_color/{recording_name}') and
                  x.endswith('pv.txt')))).split('_')[0]
    image_paths = sorted([
        x for x in namelist if
        x.startswith(f'egocentric_color/{recording_name}/{datetime}/PV/') and
        x.endswith('jpg')])
    matches = [
        re.match(r'(?P<timestamp>\d+)_frame_(?P<frame_id>\d+)\.jpg', osp.basename(x))
        for x in image_paths]
    timestamp_to_frame_id = {m['timestamp']: int(m['frame_id']) for m in matches}
    frame_id_to_image_path = {int(m['frame_id']): p for m, p in zip(matches, image_paths)}

    lines = zipf_ego.open(
        f'egocentric_color/{recording_name}/{datetime}/{datetime}_pv.txt').read().decode(
        'utf8').splitlines()
    cx, cy, w, h = map(float, lines[0].split(','))
    frame_id_to_camera = {}

    for i, calib_of_frame in enumerate(lines[1:]):
        calib_of_frame = calib_of_frame.split(',')
        timestamp = calib_of_frame[0]
        if timestamp in timestamp_to_frame_id:
            fx = float(calib_of_frame[1])
            fy = float(calib_of_frame[2])
            holo_world_to_ego = np.linalg.inv(
                np.array(calib_of_frame[3:20]).astype(float).reshape((4, 4)))
            world_to_ego = holo_world_to_ego @ world_to_holo_world
            cam = cameralib.Camera(
                extrinsic_matrix=np.reshape([1, -1, -1, 1], [-1, 1]) * world_to_ego,
                intrinsic_matrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                distortion_coeffs=np.zeros(5), world_up=(0, 1, 0))
            frame_id_to_camera[timestamp_to_frame_id[timestamp]] = cam

    return frame_id_to_camera, frame_id_to_image_path


if __name__ == '__main__':
    main()
