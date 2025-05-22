import argparse
import csv
import itertools
import json
import os.path as osp

import boxlib
import cameralib
import ezc3d
import imageio.v2 as imageio
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache('aspset_stage1.pkl', min_time="2021-12-05T21:47:55")
def make_stage1():
    aspset_root = f'{DATA_ROOT}/aspset/data'
    seq_train, seq_val, seq_test = load_split(f'{aspset_root}/splits.csv')

    def process_sequences(seqs, direc, pool):
        examples = []
        for subj_id, vid_id, view in spu.progressbar(seqs):
            boxes = load_boxes(
                f'{aspset_root}/{direc}/boxes/{subj_id}/{subj_id}-{vid_id}-{view}.csv')
            camera = load_camera(f'{aspset_root}/{direc}/cameras/{subj_id}/{subj_id}-{view}.json')
            video_path = f'{aspset_root}/{direc}/videos/{subj_id}/{subj_id}-{vid_id}-{view}.mkv'
            joints = load_joints(
                f'{aspset_root}/{direc}/joints_3d/{subj_id}/{subj_id}-{vid_id}.c3d')

            video = imageio.get_reader(video_path)
            video_relpath = osp.relpath(video_path, aspset_root)
            pose_sampler = AdaptivePoseSampler(100)
            for i_frame, (world_pose, frame, box) in enumerate(zip(joints, video, boxes)):
                if pose_sampler.should_skip(world_pose):
                    continue
                new_image_relpath = f'aspset_downscaled/{video_relpath}/{i_frame:06d}.jpg'
                ex = ds3d.Pose3DExample(frame, world_pose, box, camera)
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath), callback=examples.append)
        return examples

    with spu.ThrottledPool() as pool:
        examples_train = process_sequences(seq_train, 'trainval', pool)
        examples_val = process_sequences(seq_val, 'trainval', pool)

    joint_info = JointInfo(
        joints='rank,rkne,rhip,rwri,relb,rsho,lank,lkne,lhip,lwri,lelb,lsho,htop,head,neck,spin,'
               'pelv',
        edges='htop-head-neck-rsho-relb-rwri,neck-spin-pelv-rhip-rkne-rank')
    return ds3d.Pose3DDataset(joint_info, examples_train, examples_val)


@spu.picklecache('aspset.pkl', min_time="2022-02-08T19:46:07")
def make_dataset():
    ds = make_stage1()
    ds3d.add_masks(ds, f'{DATA_ROOT}/aspset_downscaled/masks', 5)
    return ds


def load_boxes(path):
    with open(path) as f:
        reader = csv.reader(f)
        ltrb = np.array([
            list(map(float, row))
            for row in itertools.islice(reader, 1, None)])
        boxes = np.concatenate([ltrb[:, :2], ltrb[:, 2:] - ltrb[:, :2]], axis=-1)
        boxes = np.array([boxlib.expand(box, 1.1) for box in boxes])
        return boxes.astype(np.float32)


def load_split(path):
    sequences_train = []
    sequences_val = []
    sequences_test = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[2] == 'train':
                target = sequences_train
            elif row[2] == 'val':
                target = sequences_val
            else:
                target = sequences_test
            views = 'left mid right'.split() if row[3] == 'all' else [row[3]]
            for view in views:
                target.append((row[0], row[1], view))
    return sequences_train, sequences_val, sequences_test


def load_camera(path):
    with open(path) as f:
        a = json.load(f)
    extr = np.array(a['extrinsic_matrix']).reshape(4, 4)
    intr = np.array(a['intrinsic_matrix']).reshape(3, 4)[:, :3]
    return cameralib.Camera(extrinsic_matrix=extr, intrinsic_matrix=intr, world_up=(0, -1, 0))


def load_joints(path):
    c3d = ezc3d.c3d(path)
    return c3d['data']['points'].transpose(2, 1, 0)[..., :3].astype(np.float32)


if __name__ == '__main__':
    main()
