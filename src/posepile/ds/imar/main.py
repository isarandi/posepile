import argparse
import glob
import os.path as osp

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import tensorflow_hub as tfhub
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument(
        '--detector-path', type=str,
        default='https://github.com/isarandi/tensorflow-yolov4-tflite/'
                'releases/download/v0.1.0/yolov4_416.tar.gz')
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1(FLAGS.dataset)
    elif FLAGS.stage == 2:
        make_dataset(FLAGS.dataset)


@spu.picklecache('imar_stage1.pkl', min_time="2022-08-01T17:20:38")
def make_stage1(name):
    joint_names = (
        'pelv,lhip,lkne,lank,rhip,rkne,rank,spin,neck,head,htop,lsho,lelb,lwri,rsho,relb,rwri,'
        'lfoo,ltoe,rfoo,rtoe,lthu,lfin,rthu,rfin')
    edges = ('htop-head-neck-spin-pelv-lhip-lkne-lank-lfoo-ltoe,'
             'lthu-lwri-lelb-lsho-neck-rsho-relb-rwri-rthu,rwri-rfin,lwri-lfin,'
             'pelv-rhip-rkne-rank-rfoo-rtoe')
    joint_info = JointInfo(joint_names, edges)

    cam_ids = ['50591643', '58860488', '60457274', '65906101']
    root = f'{DATA_ROOT}/imar_datasets/{name}'
    ds_info = spu.load_json(f'{root}/{name}_info.json')

    detector = tfhub.load(FLAGS.detector_path)
    examples = []
    train_subjects = ds_info['train_subj_names']

    with spu.ThrottledPool() as pool:
        for subj, seq_names in ds_info['subj_to_act'].items():
            print(subj, train_subjects)
            if subj not in train_subjects:
                continue
            for seq_name in seq_names:
                print(seq_name)
                video_world_coords = load_coords(f'{root}/train/{subj}/joints3d_25/{seq_name}.json')

                for cam_id in cam_ids:
                    print(cam_id)
                    video_path = f'{root}/train/{subj}/videos/{cam_id}/{seq_name}.mp4'
                    camera = load_camera(
                        f'{root}/train/{subj}/camera_parameters/{cam_id}/{seq_name}.json')
                    pose_sampler = AdaptivePoseSampler2(100, True, True, 10000)

                    with imageio.get_reader(video_path, 'ffmpeg', fps=200) as frames:
                        for i_frame, (frame, world_coords) in enumerate(
                                zip(frames, video_world_coords)):
                            if pose_sampler.should_skip(world_coords):
                                continue

                            detections = detector.predict_single_image(frame, 0.2, 0.7).numpy()
                            if detections.size == 0:
                                continue

                            imcoords = camera.world_to_image(world_coords)
                            gt_box = boxlib.expand(boxlib.bb_of_points(imcoords), 1.2)
                            ious = [boxlib.iou(gt_box, det[:4]) for det in detections]
                            i_det = np.argmax(ious)

                            if ious[i_det] < 0.5:
                                continue
                            box = detections[i_det][:4]
                            new_image_replath = (
                                    f'{name}_downscaled/' + osp.relpath(video_path, root) +
                                    f'/{i_frame:06d}.jpg')
                            ex = ds3d.Pose3DExample(frame, world_coords, bbox=box, camera=camera)
                            pool.apply_async(
                                make_efficient_example, (ex, new_image_replath),
                                callback=examples.append)

    return ds3d.Pose3DDataset(joint_info, train_examples=examples)


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('imar.pkl', min_time="2022-07-28T12:25:27")
def make_dataset(name):
    ds = make_stage1(name)

    mask_paths = glob.glob(f'{DATA_ROOT}/{name}_downscaled/masks/*.pkl')
    mask_dict = {}
    for path in mask_paths:
        mask_dict.update(spu.load_pickle(path))

    for ex in [*ds.examples[0], *ds.examples[1]]:
        relpath = spu.last_path_components(ex.image_path, 6)
        ex.mask = mask_dict[relpath]
    return ds


def load_coords(path):
    return np.array(spu.load_json(path)['joints3d_25'], np.float32) * 1000


def load_camera(path):
    calib = spu.load_json(path)
    R = np.array(calib['extrinsics']['R'], np.float32)
    t = np.array(calib['extrinsics']['T'][0], np.float32) * 1000
    intr = calib['intrinsics_w_distortion']
    f = np.array(intr['f'][0], np.float32)
    c = np.array(intr['c'][0], np.float32)
    k = np.array(intr['k'][0], np.float32)
    p = np.array(intr['p'][0], np.float32)
    intrinsic_matrix = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], np.float32)
    dist_coeffs = np.array([k[0], k[1], p[0], p[1], k[2]], np.float32)
    return cameravision.Camera(
        rot_world_to_cam=R, optical_center=t, intrinsic_matrix=intrinsic_matrix,
        distortion_coeffs=dist_coeffs, world_up=(0, 0, 1))


if __name__ == '__main__':
    main()
