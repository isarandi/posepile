import os
import os.path as osp

import numpy as np
import simplepyutils as spu

from posepile.ds.experimental.cwi.triangulate import triangulate_poses
from posepile.ds.experimental.triangulate_common import mask_and_average
from posepile.paths import DATA_ROOT

MPI08_ROOT = f'{DATA_ROOT}/mpi08'


def main():
    video_paths_all = spu.sorted_recursive_glob(f'{MPI08_ROOT}/*/*/*.avi')
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    video_paths = list(spu.groupby(video_paths_all, osp.dirname).values())[i_task]
    print(video_paths)
    video_relpaths = [osp.relpath(p, MPI08_ROOT) for p in video_paths]

    out_path = f'{MPI08_ROOT}/triang/{osp.dirname(video_relpaths[0])}/output.pkl'
    if spu.is_pickle_readable(out_path):
        return

    cameras_all = spu.load_pickle(f'{MPI08_ROOT}/cameras.pkl')
    cameras = [cameras_all[p] for p in video_relpaths]
    preds = [spu.load_pickle(spu.replace_extension(f'{MPI08_ROOT}/pred/{p}', '.pkl'))
             for p in video_relpaths]
    poses3d = [p['poses3d'] for p in preds]

    nan = np.full([5, 122, 3], dtype=np.float32, fill_value=np.nan)

    triangs = []
    for world_poses_per_cam in spu.progressbar(zip(*poses3d), total=len(poses3d[0])):
        world_pose_per_cam = [w[0] if len(w) > 0 else nan for w in world_poses_per_cam]
        campose_per_cam = np.array([
            c.world_to_camera(p) for c, p in zip(cameras, world_pose_per_cam)])
        campose_per_cam = mask_and_average(campose_per_cam)
        triang = triangulate_poses(cameras, campose_per_cam)
        triangs.append(triang)

    triangs = np.array(triangs)
    spu.dump_pickle(triangs, out_path)


if __name__ == '__main__':
    main()
