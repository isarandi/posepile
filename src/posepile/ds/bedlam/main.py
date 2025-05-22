import os
import os.path as osp
import re

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import rlemasklib
import scipy.optimize
import simplepyutils as spu
import smpl
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import logger
import smpl.tensorflow
np.str = str


def main():
    dataset = make_dataset()


@spu.picklecache('bedlam2.pkl', min_time="2023-10-23T23:09:02")
def make_dataset():
    print('Staring')
    bedlam_root = f'{DATA_ROOT}/bedlam'
    anno_filenames = sorted(os.listdir(f'{bedlam_root}/all_npz_12_training/'))
    group_names = ['_'.join(osp.splitext(f)[0].split('_')[:-1])
                   for f in anno_filenames if f.startswith('2022')]
    print(f'Found {len(group_names)} groups')
    body_model_smplx = smpl.tensorflow.SMPL(
        model_root=f'{DATA_ROOT}/body_models/smplx', gender='neutral', model_name='smplx')
    # body_model_smpl = smpl.SMPL(
    #    model_root=f'{DATA_ROOT}/body_models/smpl', gender='neutral', model_name='smpl')

    joint_info = ds3d.JointInfo([], [])
    examples = []
    with spu.ThrottledPool() as pool:
        for group_name in group_names:
            try:
                annos_smplx = np.load(f'{bedlam_root}/all_npz_12_training/{group_name}_6fps.npz')
                # annos_smpl = np.load(
                #    f'{bedlam_root}/all_npz_12_smpl_training/{group_name}_6fps.npz')
            except FileNotFoundError:
                annos_smplx = np.load(f'{bedlam_root}/all_npz_12_training/{group_name}_30fps.npz')
                # annos_smpl = np.load(
                #    f'{bedlam_root}/all_npz_12_smpl_training/{group_name}_30fps.npz')
            imgnames = list(annos_smplx['imgname'])

            # poses_smpl = list(annos_smpl['pose_world'])
            # shapes_smpl = list(annos_smpl['shape'])
            # transs_smpl = list(annos_smpl['trans_world'])

            poses_smplx = list(annos_smplx['pose_world'])
            shapes_smplx = list(annos_smplx['shape'])
            transs_smplx = list(annos_smplx['trans_world'])
            cam_ints = list(annos_smplx['cam_int'])
            cam_exts = list(annos_smplx['cam_ext'])
            jointss_smplx = body_model_smplx(
                annos_smplx['pose_world'], annos_smplx['shape'],
                annos_smplx['trans_world'],
                return_vertices=False)['joints'] * 1000

            i_seq_to_indices = spu.groupby(
                range(len(imgnames)),
                lambda i: int(imgnames[i].split('/')[0].split('_')[1]))

            for i_seq, indices_of_seq in spu.progressbar_items(i_seq_to_indices):
                video_path = f'{bedlam_root}/{group_name}/mp4/seq_{i_seq:06d}.mp4'
                detections_allframes = spu.load_pickle(
                    f'{bedlam_root}/detections/{group_name}/mp4/seq_{i_seq:06d}.pkl')

                pose_sampler = AdaptivePoseSampler2(
                    100, True, True, 100)

                i_frame_to_indices = spu.groupby(
                    indices_of_seq,
                    lambda i: int(
                        re.match(r'seq_\d+/seq_\d+_(?P<i_frame>\d+).png',
                                 imgnames[i])['i_frame']))

                with imageio.get_reader(
                        video_path, 'ffmpeg', output_params=['-map', '0:v:0']) as reader:

                    for i_frame, frame in enumerate(reader):
                        indices_of_frame = i_frame_to_indices.get(i_frame)
                        if indices_of_frame is None:
                            continue

                        mask = imageio.imread(
                            f'{bedlam_root}/{group_name}/masks/seq_{i_seq:06d}/'
                            f'seq_{i_seq:06d}_{i_frame:04d}_env.png')
                        if 'closeup' in group_name:
                            mask = np.rot90(mask, -1)

                        mask = rlemasklib.encode(mask == 0)

                        gt_people = []
                        gt_anno_ids = []
                        for i_ann in indices_of_frame:
                            imgname = imgnames[i_ann]
                            m = re.match(r'seq_\d+/seq_(?P<i_seq>\d+)_(?P<i_frame>\d+).png',
                                         imgname)
                            assert i_frame == int(m['i_frame'])
                            assert i_seq == int(m['i_seq'])

                            joints_smplx = jointss_smplx[i_ann]

                            # joints_smpl = body_model_smpl(
                            #     poses_smpl[i_ann][np.newaxis], shapes_smpl[i_ann][np.newaxis],
                            #     transs_smpl[i_ann][np.newaxis],
                            #     return_vertices=False)['joints'][0] * 1000
                            camera = cameravision.Camera(
                                intrinsic_matrix=cam_ints[i_ann],
                                extrinsic_matrix=cam_exts[i_ann], world_up=(0, -1, 0))
                            camera.t *= 1000
                            if pose_sampler.should_skip(camera.world_to_camera(joints_smplx)):
                                continue

                            bbox = boxlib.expand(
                                boxlib.bb_of_points(camera.world_to_image(joints_smplx)),
                                1.05)

                            parameters = dict(
                                type='smplx',
                                gender='neutral', pose=poses_smplx[i_ann],
                                shape=shapes_smplx[i_ann],
                                kid_factor=np.float32(0), trans=transs_smplx[i_ann])

                            ex = ds3d.Pose3DExample(
                                image_path=frame, world_coords=np.zeros((0, 3), np.float32),
                                bbox=bbox, camera=camera, mask=mask, parameters=parameters)
                            gt_people.append(ex)
                            gt_anno_ids.append(i_ann)

                        if not gt_people:
                            logger.info(f'No GT people')
                            continue

                        boxes = get_boxes(
                            gt_boxes=[ex.bbox for ex in gt_people],
                            detections=detections_allframes[i_frame])

                        for ex, box, gt_anno_id in zip(gt_people, boxes, gt_anno_ids):
                            if box is None or min(box[2:4]) < 100:
                                continue

                            ex.bbox = box
                            new_image_relpath = f'bedlam_downscaled/{group_name}/seq_{i_seq:06d}/' \
                                                f'seq_{i_seq:06d}_{i_frame:04d}_{gt_anno_id}.jpg'
                            pool.apply_async(
                                make_efficient_example, (ex, new_image_relpath),
                                dict(min_time="2023-10-23T22:09:51", assume_image_ok=True),
                                callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, examples)


def get_boxes(gt_boxes, detections, iou_thresh=0.5):
    if detections.size == 0:
        return gt_boxes
    iou_matrix = np.array([[boxlib.iou(gt_box[:4], det[:4])
                            for det in detections]
                           for gt_box in gt_boxes])
    gt_indices, det_indices = scipy.optimize.linear_sum_assignment(
        -iou_matrix)
    result_boxes = [None for b in gt_boxes]
    for i_gt, i_det in zip(gt_indices, det_indices):
        if iou_matrix[i_gt, i_det] >= iou_thresh:
            result_boxes[i_gt] = detections[i_det][:4]
    return result_boxes


if __name__ == '__main__':
    main()
