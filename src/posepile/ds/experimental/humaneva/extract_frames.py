import os
import os.path as osp

import cv2
import imageio.v2 as imageio
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    video_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/humaneva/**/*.avi')
    spu.parallel_map_with_progbar(extract_frames, video_paths)


def extract_frames(src_video_path):
    dst_folder_path = osp.splitext(src_video_path)[0]
    os.makedirs(dst_folder_path, exist_ok=True)
    for i_frame, frame in enumerate(video_frames(src_video_path)):
        dst_path = f'{dst_folder_path}/frame_{i_frame:06d}.jpg'
        imageio.imwrite(dst_path, frame, quality=95)


def video_frames(path):
    video = None
    try:
        video = cv2.VideoCapture(path)
        while True:
            ret, frame = video.read()
            if not ret:
                return
            yield frame[..., ::-1]
    finally:
        if video is not None:
            video.release()


if __name__ == '__main__':
    main()
