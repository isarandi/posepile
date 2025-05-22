import multiprocessing
import os
import os.path as osp
import pathlib

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    pool = multiprocessing.Pool()
    bbox_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/h36m/**/ground_truth_bb/*.mat')
    spu.parallel_map_with_progbar(extract_bounding_boxes, bbox_paths, pool)

    video_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/h36m/**/Videos/*.mp4')
    spu.parallel_map_with_progbar(extract_frames, video_paths, pool)


def extract_bounding_boxes(src_matfile_path):
    """Human3.6M supplies bounding boxes in the form of masks with 1s inside the box and 0s
    outside. This converts from that format to NumPy files containing the bounding box coordinates
    in [left, top, width, height] representation.
    """
    with h5py.File(src_matfile_path, 'r') as f:
        refs = f['Masks'][:, 0]
        bboxes = np.empty([len(refs), 4], dtype=np.float32)
        for i, ref in enumerate(refs):
            mask = np.array(f['#refs#'][ref]).T
            bboxes[i] = np.array(cv2.boundingRect(mask), np.float32)

    filename = osp.basename(src_matfile_path)
    dst_file_path = pathlib.Path(src_matfile_path).parents[2] / f'BBoxes/{filename[:-4]}.npy'
    spu.ensure_parent_dir_exists(dst_file_path)
    np.save(dst_file_path, bboxes)


def extract_frames(src_video_path, every_nth=(5, 64)):
    """Save every 5th and 64th frame from a video as images."""
    video_name = spu.path_stem(src_video_path)
    dst_folder_path = pathlib.Path(src_video_path).parents[1] / 'Images' / video_name
    os.makedirs(dst_folder_path, exist_ok=True)

    with imageio.get_reader(src_video_path, 'ffmpeg') as reader:
        for i_frame, frame in enumerate(reader):
            if any(i_frame % x == 0 for x in every_nth):
                dst_filename = f'frame_{i_frame:06d}.jpg'
                dst_path = osp.join(dst_folder_path, dst_filename)
                imageio.imwrite(dst_path, frame, quality=95)


if __name__ == '__main__':
    main()
