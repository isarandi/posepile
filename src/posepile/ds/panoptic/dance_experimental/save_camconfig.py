import os.path as osp

import simplepyutils as spu

import posepile.ds.panoptic.main as panoptic_main
from posepile.paths import DATA_ROOT


def main():
    seq_names = (
            [f'160317_moonbaby{i}' for i in range(1, 4)] +
            [f'150821_dance{i}' for i in range(1, 6)])
    root = f'{DATA_ROOT}/panoptic'

    all_cameras = {}
    for seq_name in seq_names:
        seq_dir = f'{root}/{seq_name}'

        if 'dance' in seq_name:
            camera_type = 'hd'
            camera_names = [f'00_{i:02d}' for i in range(0, 30) if i != 23]
            cameras = panoptic_main.get_cameras(
                f'{seq_dir}/calibration_{seq_name}_corrected_full.json', camera_names)
            video_paths = [
                f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}_undistorted.mp4'
                for cam_name in camera_names]
        else:
            camera_type = 'kinect'
            camera_names = [f'50_{i:02d}' for i in range(1, 11)]
            cameras = panoptic_main.get_cameras(
                f'{seq_dir}/calibration_{seq_name}.json', camera_names)
            video_paths = [
                f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}_undistorted.mp4'
                for cam_name in camera_names]

        cameras = [undistort_camera(cameras[name]) for name in camera_names]
        video_relpaths = [osp.relpath(p, root) for p in video_paths]
        all_cameras.update(zip(video_relpaths, cameras))

    print(all_cameras)
    spu.dump_pickle(all_cameras, f'{root}/dance_cameras.pkl')


def undistort_camera(camera):
    undist_camera = camera.copy()
    undist_camera.undistort()
    undist_camera.square_pixels()
    return undist_camera


if __name__ == '__main__':
    main()
