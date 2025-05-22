import argparse
import os
import os.path as osp

import boxlib
import cameralib
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import smpl.numpy
from simplepyutils import FLAGS
from posepile.ds.rich.add_parametric import load_smplx_params
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

DATASET_NAME = 'intercap'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_dataset_unsegmented()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache(f'{DATASET_NAME}_unsegmented.pkl', min_time="2023-12-01T13:35:22")
def make_dataset_unsegmented():
    detections_all = spu.load_pickle(f'{DATASET_DIR}/yolov4_detections.pkl')
    genders = [
        'male', 'male', 'female', 'male', 'male', 'female', 'female', 'female', 'female', 'male']
    examples = []
    broken_image_relpaths = [
        'RGBD_Individuals/07/04/Seg_1/Frames_Cam3/color/00170.jpg',
        'RGBD_Individuals/09/08/Seg_0/Frames_Cam2/color/00078.jpg']

    with spu.ThrottledPool() as pool:
        for i_subject in range(1, 11):
            i_motions = [int(x) for x in
                         os.listdir(f'{DATASET_DIR}/Res_Individuals/{i_subject:02d}')]
            gender = genders[i_subject - 1]
            bm = smpl.numpy.get_cached_body_model('smplx', gender)
            for i_motion in i_motions:
                cameras = get_cameras(i_subject, i_motion)
                segs = os.listdir(f'{DATASET_DIR}/Res_Individuals/{i_subject:02d}/{i_motion:02d}')
                for seg in segs:
                    data = spu.load_pickle(
                        f'{DATASET_DIR}/Res_Individuals/{i_subject:02d}/{i_motion:02d}/'
                        f'{seg}/res_1.pkl')

                    sampler = AdaptivePoseSampler2(
                        100, True, True, 100)
                    for i_frame, frame_data in enumerate(spu.progressbar(
                            data, desc=f'{i_subject:02d}/{i_motion:02d}/{seg}')):
                        pose, shape, trans, kid_factor, expression = load_smplx_params(
                            frame_data, gender)
                        joints3d = bm.single(
                            pose, shape, trans, return_vertices=False)['joints'] * 1000
                        if sampler.should_skip(joints3d):
                            continue

                        parameters = dict(
                            type='smplx', gender=gender, pose=pose, shape=shape, trans=trans)

                        for i_cam, cam in enumerate(cameras):
                            relpath = (f'RGBD_Individuals/{i_subject:02d}/{i_motion:02d}/'
                                       f'{seg}/Frames_Cam{i_cam + 1}/color/{i_frame:05d}.jpg')
                            if relpath in broken_image_relpaths:
                                continue

                            impath = f'{DATASET_DIR}/{relpath}'
                            if not osp.exists(impath):
                                print(f'Image not found: {impath}')
                                continue

                            detections = detections_all[relpath]
                            if len(detections) == 0:
                                continue
                            bbox_gt = boxlib.expand(
                                boxlib.bb_of_points(cam.world_to_image(joints3d)), 1.1)
                            ious = [boxlib.iou(det, bbox_gt) for det in detections]
                            if np.max(ious) < 0.1:
                                continue

                            bbox_det = detections[np.argmax(ious)][:4]

                            ex = ds3d.Pose3DExample(
                                image_path=impath, camera=cam, parameters=parameters,
                                world_coords=None, bbox=bbox_det)
                            new_image_relpath = f'{DATASET_NAME}_downscaled/{relpath}'
                            pool.apply_async(
                                make_efficient_example, (ex, new_image_relpath),
                                callback=examples.append)

    return ds3d.Pose3DDataset(
        ds3d.JointInfo([], []), examples)


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    ds = make_dataset_unsegmented()
    ds3d.add_masks(
        ds, f'{DATA_ROOT}/{DATASET_NAME}_downscaled/masks',
        relative_root=f'{DATA_ROOT}/{DATASET_NAME}_downscaled')
    return ds


def get_cameras(subject, motion):
    focals = np.array(
        [[918.457763671875, 918.4373779296875], [915.29962158203125, 915.1966552734375],
         [912.8626708984375, 912.67633056640625], [909.82025146484375, 909.62469482421875],
         [920.533447265625, 920.09722900390625], [909.17633056640625, 909.23529052734375]])
    centers = np.array(
        [[956.9661865234375, 555.944580078125], [956.664306640625, 551.6165771484375],
         [956.72003173828125, 554.2166748046875], [957.6181640625, 554.60296630859375],
         [958.4615478515625, 550.42987060546875], [956.14801025390625, 555.01593017578125]])
    distortions = np.array(
        [[0.535593, -2.509073, 0.000718, -0.000244, 1.362741, 0.414365, -2.340596, 1.297858],
         [0.486854, -2.639548, 0.000848, -0.000512, 1.499817, 0.363917, -2.457485, 1.424830],
         [0.457903, -2.524319, 0.000733, -0.000318, 1.464439, 0.340047, -2.355746, 1.395222],
         [0.396468, -2.488340, 0.000909, -0.000375, 1.456987, 0.278806, -2.316228, 1.385524],
         [0.615471, -2.643317, 0.000616, -0.000661, 1.452086, 0.492699, -2.474038, 1.386289],
         [0.494798, -2.563026, 0.000720, -0.000212, 1.484987, 0.376524, -2.396207, 1.416732]])
    intrinsic_matrices = [np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0., 0., 1.]]) for f, c in
                          zip(focals, centers)]
    extrinsics = get_extrinsic_matrices(subject, motion)
    result = [cameralib.Camera(
        intrinsic_matrix=intr, extrinsic_matrix=extr, distortion_coeffs=dist, world_up=(0, -1, 0))
        for intr, extr, dist in zip(intrinsic_matrices, extrinsics, distortions)]
    for c in result:
        c.t *= 1000

    world_up_vector = np.sum([-c.R[1] for c in result], axis=0)
    world_up_vector /= np.linalg.norm(world_up_vector)
    for c in result:
        c.world_up = world_up_vector
    return result


def get_extrinsic_matrices(subject, motion):
    mats = [
        np.eye(4),
        np.array(
            [[0.578526, -0.198473, 0.791148, -1.98866],
             [0.16954, 0.97802, 0.121377, -0.254528],
             [-0.797849, 0.0639117, 0.59946, 1.13892],
             [0, 0, 0, 1],
             ]),
        np.array(
            [[-0.448919, -0.156267, 0.879802, -2.2178],
             [0.133952, 0.961696, 0.239161, -0.597409],
             [-0.883475, 0.225215, -0.410791, 3.5759],
             [0, 0, 0, 1]
             ]),
        np.array(
            [[-0.997175, -0.0640807, 0.0391773, 0.0546731],
             [-0.0501059, 0.956143, 0.288582, -0.806246],
             [-0.0559517, 0.285804, -0.956653, 5.28593],
             [0, 0, 0, 1],
             ]),
        np.array(
            [[-0.410844, 0.0626555, -0.90955, 2.65605],
             [-0.188877, 0.970143, 0.152145, -0.45593],
             [0.891926, 0.234302, -0.386743, 3.65228],
             [0, 0, 0, 1]
             ]),
        np.array(
            [[0.424204, 0.0773163, -0.90226, 2.32111],
             [-0.0993286, 0.994309, 0.0385042, -0.106149],
             [0.900103, 0.0732864, 0.42947, 1.05865],
             [0, 0, 0, 1]
             ])]
    subject = int(subject)
    motion = int(motion)
    if (subject in [8, 9, 10]) or (subject in [1, 2, 4, 5, 6, 7] and motion in [8, 9, 10]):
        mats[1] = np.array(
            [[0.586004, -0.197023, 0.78599, -2.0087],
             [0.167134, 0.978521, 0.120675, -0.268243],
             [-0.792884, 0.0606496, 0.606348, 1.14359],
             [0, 0, 0, 1]
             ])
        mats[2] = np.array(
            [[-0.433389, -0.155316, 0.887722, -2.21588],
             [0.137053, 0.962221, 0.235259, -0.597695],
             [-0.890724, 0.223623, -0.39573, 3.59858],
             [0, 0, 0, 1]
             ])
    elif (subject == 3 and motion in [8, 9, 10]) or (
            subject in [5, 6, 7] and motion in [1, 2, 3, 4, 5, 6, 7]):
        mats[2] = np.array(
            [[-0.433389, -0.155316, 0.887722, -2.21588],
             [0.137053, 0.962221, 0.235259, -0.597695],
             [-0.890724, 0.223623, -0.39573, 3.59858],
             [0, 0, 0, 1]
             ])
    return [np.linalg.inv(m).astype(np.float32) for m in mats]


if __name__ == '__main__':
    main()
