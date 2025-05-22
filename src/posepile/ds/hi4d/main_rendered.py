import glob
import os.path as osp

import boxlib
import numpy as np
import posepile.datasets3d as ds3d
import scipy.optimize
import simplepyutils as spu
from posepile.paths import DATA_ROOT


def main():
    make_dataset()


@spu.picklecache('hi4d_rendered.pkl', min_time="2023-12-19T22:23:15")
def make_dataset():
    examples = []
    detections_all = spu.load_pickle(f'{DATA_ROOT}/hi4d_rerender/yolov4_detections.pkl')
    camera_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/hi4d_rerender/**/cameras.pkl')

    for camera_path in spu.progressbar(camera_paths):
        seq_id = spu.path_range(camera_path, -3, -1)
        metadata = np.load(f'{DATA_ROOT}/hi4d/{seq_id}/meta.npz')
        cameras = spu.load_pickle(camera_path)
        masks = spu.load_pickle(f'{DATA_ROOT}/hi4d_rerender/{seq_id}/masks.pkl')

        frame_paths = sorted(glob.glob(f'{DATA_ROOT}/hi4d_rerender/{seq_id}/f_*.jpg'))
        i_gen_to_i_frame_i_person = {}
        for frame_path in frame_paths:
            base_noext = osp.splitext(osp.basename(frame_path))[0]
            parts = base_noext.split('_')
            i_gen = int(parts[1])
            i_frame = int(parts[2])
            i_person = int(parts[3])
            i_gen_to_i_frame_i_person[i_gen] = (i_frame, i_person)


        for i_gen, (camera, mask) in enumerate(zip(cameras, masks)):
            i_frame, i_person = i_gen_to_i_frame_i_person[i_gen]
            impath = (f'{DATA_ROOT}/hi4d_rerender/'
                      f'{seq_id}/f_{i_gen:06d}_{i_frame:06d}_{i_person}.jpg')
            smpl_data = np.load(f'{DATA_ROOT}/hi4d/{seq_id}/smpl/{i_frame:06d}.npz')
            pose = np.concatenate(
                [smpl_data['global_orient'], smpl_data['body_pose']], axis=-1)
            image_relpath = osp.relpath(impath, f'{DATA_ROOT}/hi4d_rerender')
            detections = detections_all[image_relpath][:, :4]
            if len(detections) == 0:
                continue

            smpl_verts = smpl_data['verts'] * 1000
            smpl_verts2d = camera.world_to_image(smpl_verts)
            boxes_gt = [boxlib.bb_of_points(s) for s in smpl_verts2d]

            iou_matrix = np.array(
                [[boxlib.iou(b1, b2) for b2 in detections] for b1 in boxes_gt])
            gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
            x = np.argwhere(gt_indices == i_person).squeeze(-1)

            if len(x) == 0:
                continue
            else:
                x = x.item()

            i_det = det_indices[x]
            if iou_matrix[i_person, i_det] < 0.1:
                continue

            joints = smpl_data['joints_3d'][i_person] * 1000
            joints2d = camera.world_to_image(joints)
            is_in_det = boxlib.contains(detections[i_det], joints2d)
            if np.sum(is_in_det) < 6:
                continue

            parameters = dict(
                type='smpl', gender=metadata['genders'][i_person], pose=pose[i_person],
                shape=smpl_data['betas'][i_person], trans=smpl_data['transl'][i_person])
            ex = ds3d.Pose3DExample(
                image_path=f'hi4d_rerender/{image_relpath}',
                camera=camera, bbox=detections[i_det],
                parameters=parameters, world_coords=None, mask=mask)
            examples.append(ex)

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples)


if __name__ == '__main__':
    main()
