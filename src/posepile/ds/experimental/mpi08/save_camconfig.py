import os
import os.path as osp

import cameralib
import numpy as np
import simplepyutils as spu

from posepile.paths import DATA_ROOT

MPI08_ROOT = f'{DATA_ROOT}/mpi08'


def main():
    cameras = [
        load_camera(f'{MPI08_ROOT}/InputFiles/ProjectionMatrices/proj{i:02d}.dat')
        for i in range(8)]

    camera_per_video = {}
    for video_path in spu.sorted_recursive_glob(f'{MPI08_ROOT}/*/*/*.avi'):
        video_relpath = osp.relpath(video_path, MPI08_ROOT)
        i_cam = int(video_path[-6:-4])
        if i_cam == 99:
            continue
        camera_per_video[video_relpath] = cameras[i_cam]

    spu.dump_pickle(camera_per_video, f'{MPI08_ROOT}/cameras.pkl')


def load_camera(path):
    lines = spu.read_lines(path)
    intr_lines = lines[6:9]
    intr = np.array([[float(x) for x in line.split()] for line in intr_lines], np.float32)
    extr_lines = lines[11:15]
    extr = np.array([[float(x) for x in line.split()] for line in extr_lines], np.float32)
    return cameralib.Camera(intrinsic_matrix=intr, extrinsic_matrix=extr, world_up=(0, 1, 0))


if __name__ == '__main__':
    main()
