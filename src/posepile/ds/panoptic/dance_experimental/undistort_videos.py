import os
import os.path as osp

import cameralib
import posepile.ds.panoptic.main as panoptic_main
import posepile.util.videoproc as videoproc
from posepile.paths import DATA_ROOT

os.environ['OMP_NUM_THREADS'] = '1'


def main():
    root = f'{DATA_ROOT}/panoptic'
    seq_names = ([f'150821_dance{i}' for i in range(1, 6)] +
                 [f'160317_moonbaby{i}' for i in range(1, 4)])

    tasks = []
    for seq_name in seq_names:
        seq_dir = f'{root}/{seq_name}'
        if 'dance' in seq_name:
            camera_type = 'hd'
            camera_names = [f'00_{i:02d}' for i in range(0, 30) if i != 23]
            video_paths = [f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}.mp4'
                           for cam_name in camera_names]
            cameras = panoptic_main.get_cameras(
                f'{seq_dir}/calibration_{seq_name}_corrected_full.json', camera_names)
        else:
            camera_type = 'kinect'
            camera_names = [f'50_{i:02d}' for i in range(1, 11)]
            video_paths = [f'{seq_dir}/{camera_type}Videos/{camera_type}_{cam_name}.mp4'
                           for cam_name in camera_names]
            cameras = panoptic_main.get_cameras(
                f'{seq_dir}/calibration_{seq_name}.json', camera_names)

        cameras = [cameras[name] for name in camera_names]
        tasks.extend(zip(video_paths, cameras))

    video_path_in, camera = tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    video_path_out = video_path_in.replace('.mp4', '_undistorted.mp4')
    undist_camera = undistort_camera(camera)
    if osp.exists(video_path_out):
        return

    videoproc.transform(
        video_path_in, video_path_out,
        lambda frame: cameralib.reproject_image(frame, camera, undist_camera, frame.shape),
        bitrate=10000000, ffmpeg_params=['-crf', '18'], macro_block_size=None)


def undistort_camera(camera):
    undist_camera = camera.copy()
    undist_camera.undistort()
    undist_camera.square_pixels()
    return undist_camera


if __name__ == '__main__':
    main()
