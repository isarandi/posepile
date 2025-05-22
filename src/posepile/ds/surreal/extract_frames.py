import os
import os.path as osp

import imageio.v2 as imageio
import simplepyutils as spu

import posepile.util.improc as improc
from posepile.paths import DATA_ROOT


def main():
    video_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/surreal/**/*.mp4')
    spu.parallel_map_with_progbar(extract_frames, video_paths)


def extract_frames(src_video_path):
    dst_folder_path = osp.splitext(src_video_path)[0]
    os.makedirs(dst_folder_path, exist_ok=True)

    with imageio.get_reader(src_video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(reader):
            dst_path = f'{dst_folder_path}/frame_{i_frame:06d}.jpg'
            if not improc.is_jpeg_readable(dst_path):
                # We flip the SURREAL images so that the coordinate system becomes right-handed
                # and works compatibly with all other datasets.
                imageio.imwrite(dst_path, frame[:, ::-1], quality=95)


if __name__ == '__main__':
    main()
