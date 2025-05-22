import os
import os.path as osp
import sys

import imageio.v2 as imageio
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    video_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/3dhp/mpi_inf_3dhp/**/video_*.avi')
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])

    src_video_path = video_paths[i_task]
    dst_folder_path = osp.dirname(src_video_path)
    video_name = spu.path_stem(src_video_path)
    i_video = int(video_name.split('_')[1])

    every_nth = int(sys.argv[1])
    with imageio.get_reader(src_video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(reader):
            if i_frame % every_nth == 0:
                dst_path = f'{dst_folder_path}/img_{i_video + 1}_{i_frame:06d}.jpg'
                if not osp.exists(dst_path):  # is_image_readable(dst_path):
                    imageio.imwrite(dst_path, frame, quality=95)


if __name__ == '__main__':
    main()
