# In NTU, it's not exactly clear which videos are taken from the same camera position
# This script looks at the actual frame content and clusters the videos
# based on some very rough frame statistics to identify if the videos are likely taken
# from the same vantage point.

# If there are issues with ffmpeg ("We had to kill ffmpeg to stop it." message),
# check out https://github.com/imageio/imageio-ffmpeg/issues/13')
# It's a timeout issue, and the solution is to increase the timeout...

import itertools
import multiprocessing
import os
import os.path as osp

import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu

from posepile.ds.experimental.ntu.main import NTU_ROOT


def main():
    video_paths_all = spu.sorted_recursive_glob(f'{NTU_ROOT}/nturgb+d_rgb/**/*_rgb.avi')
    ignores = spu.read_lines(f'{NTU_ROOT}/ignore_videos.txt')
    video_paths_all = [p for p in video_paths_all if get_video_id(p) not in ignores]
    video_paths_per_setupcam = spu.groupby(video_paths_all, lambda p: get_video_id(p)[:8])
    i_task = int(os.environ['SLURM_ARRAY_TASK_ID'])
    setupcam_name, video_paths = list(video_paths_per_setupcam.items())[i_task]
    print(setupcam_name)

    out_path = f'{NTU_ROOT}/camcalib/clustering/{setupcam_name}_max.pkl'
    # if osp.exists(out_path):
    #    return

    pool = multiprocessing.Pool()
    descriptor_path = f'{NTU_ROOT}/camcalib/clustering/descriptors_{setupcam_name}.pkl'
    if osp.exists(descriptor_path):
        descriptors = spu.load_pickle(descriptor_path)
    else:
        descriptors = pool.map(get_descriptor, video_paths)
        spu.dump_pickle(descriptors, descriptor_path)

    labels = leader_clustering(descriptors, threshold=0.5)
    video_id_to_calib_id = {
        get_video_id(path): setupcam_name + f'V{label + 1:03d}'
        for path, label in zip(video_paths, labels)}

    spu.dump_pickle(video_id_to_calib_id, out_path)


def leader_clustering(descriptors, threshold=0.5):
    clusters = []
    labels = []

    for i_item, descriptor in enumerate(descriptors):
        for i_cluster, cluster in enumerate(clusters):
            if any((dist := distance(descriptor, cluster_item)) < threshold
                   for cluster_item in cluster):
                labels.append(i_cluster)
                cluster.append(descriptor)
                break
        else:
            labels.append(len(clusters))
            clusters.append([descriptor])
    return labels


def get_descriptor(video_path, border_size=100):
    with imageio.get_reader(video_path, 'ffmpeg') as reader:
        frame_descriptors = np.stack([
            (np.stack([
                frame[:, :border_size].reshape(-1, 3),
                frame[:, -border_size:].reshape(-1, 3)],
                axis=0).astype(np.float32))
            for frame in itertools.islice(reader, 0, None, 5)], axis=0)

    x = np.median(frame_descriptors, axis=0)
    mean = np.mean(x, axis=1, keepdims=True)
    stdev = np.std(x, axis=1, keepdims=True)
    return (x - mean) / stdev


def distance(d1, d2):
    return np.sqrt(np.max(np.mean(np.square(d1 - d2), axis=(-1, -2))))


def get_video_id(path):
    return osp.basename(path).split('_')[0]


if __name__ == '__main__':
    main()
