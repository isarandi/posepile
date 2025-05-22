import re

import simplepyutils as spu

from posepile.ds.experimental.triangulate_common import *
from posepile.paths import DATA_ROOT

CHICO_ROOT = f'{DATA_ROOT}/chico'


def main():
    pred_dir = f'{CHICO_ROOT}/1c6f6193_pred'
    camera_names = ['00_03', '00_06', '00_12']

    pred_paths = spu.sorted_recursive_glob(f'{pred_dir}/**/*.pkl')
    pred_paths_per_cam = spu.groupby(pred_paths, get_camera_name_from_pred_path)
    poses_per_cam_dict = {
        cam_name: [pred_dense(spu.load_pickle(p)['poses3d'], 15) for p in spu.progressbar(paths)]
        for cam_name, paths in pred_paths_per_cam.items()}
    poses_per_cam = np.stack(
        [np.concatenate(poses_per_cam_dict[cam_name])
         for cam_name in camera_names], axis=0)[:, :, :24]
    cameras_dict = load_cameras()
    cameras = [cameras_dict[n] for n in camera_names]
    cameras[1] = calibrate(poses_per_cam[0], poses_per_cam[1], cameras[0], cameras[1])
    cameras[2] = calibrate(poses_per_cam[0], poses_per_cam[2], cameras[0], cameras[2])
    spu.dump_pickle(dict(zip(camera_names, cameras)), f'{CHICO_ROOT}/recalibrated_cameras.pkl')


def calibrate(poses_cam_ref, poses_cam, cam_ref, cam):
    relative_cam = calibrate_from_poses(poses_cam_ref, poses_cam, cam)
    return cameralib.Camera(
        intrinsic_matrix=cam.intrinsic_matrix,
        distortion_coeffs=cam.distortion_coeffs,
        extrinsic_matrix=relative_cam.get_extrinsic_matrix() @ cam_ref.get_extrinsic_matrix(),
        world_up=cam.world_up)


def pred_dense(pred3d, stdev_thresh=40):
    _, n_aug, n_joints, n_coord = pred3d[0].shape
    poses_out = np.full([len(pred3d), n_aug, n_joints, 3], fill_value=np.nan, dtype=np.float32)

    for i_frame, poses_in_frame in enumerate(pred3d):
        if len(poses_in_frame) > 0:
            poses_out[i_frame] = poses_in_frame[0]

    return mask_and_average(poses_out, stdev_thresh)


def load_cameras():
    cameras = {}
    for c in spu.load_json(f'{DATA_ROOT}/chico/camera_calib_parameters.json')['cameras']:
        intrinsic_matrix = np.array(c['K'], dtype=np.float32)
        d = np.array(c['distCoef'], dtype=np.float32)
        R = np.array(c['R'], dtype=np.float32)
        t = np.squeeze(np.array(c['t'], dtype=np.float32), 1)
        name = c['name']
        cameras[name] = cameralib.Camera(
            intrinsic_matrix=intrinsic_matrix, trans_after_rot=t, rot_world_to_cam=R.T,
            distortion_coeffs=d, world_up=[0, 0, 1])
    return cameras


def get_camera_name_from_pred_path(p):
    return re.search(r'S\d\d_.+?/(?P<camera>\d\d_\d\d)\.pkl', p)['camera']


if __name__ == '__main__':
    main()
