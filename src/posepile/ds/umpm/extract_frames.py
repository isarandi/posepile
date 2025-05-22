import os
import os.path as osp

import imageio
import simplepyutils as spu

import posepile.util.improc as improc
from posepile.paths import DATA_ROOT


def main():
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    video_paths_all = spu.sorted_recursive_glob(f'{DATA_ROOT}/umpm/Video/*.avi')
    n_video_per_task = 3
    video_paths = video_paths_all[i_task * n_video_per_task:(i_task + 1) * n_video_per_task]
    spu.parallel_map_with_progbar(extract_frames, video_paths)


def extract_frames(src_video_path):
    dst_folder_path = osp.splitext(src_video_path)[0]
    os.makedirs(dst_folder_path, exist_ok=True)

    with imageio.get_reader(src_video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(spu.progressbar(reader)):
            dst_path = f'{dst_folder_path}/frame_{i_frame:06d}.jpg'
            if not improc.is_jpeg_readable(dst_path):
                imageio.imwrite(dst_path, frame, quality=95)


if __name__ == '__main__':
    main()
