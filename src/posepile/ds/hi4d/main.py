import barecat
import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import rlemasklib
import scipy.optimize
import simplepyutils as spu
from posepile.ds.agora.add_parametric import example_to_dict
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from rlemasklib import RLEMask

def main():
    make_dataset()


@spu.picklecache('hi4d.pkl', min_time="2023-12-20T16:06:21")
def make_dataset():
    metadata_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/hi4d/*/*/meta.npz')
    examples = []
    detections_all = spu.load_pickle(f'{DATA_ROOT}/hi4d/yolov4_detections.pkl')

    with spu.ThrottledPool() as pool:
        for pbar, metadata_path in spu.zip_progressbar(metadata_paths):
            pbar.set_description(metadata_path)
            seq_id = spu.path_range(metadata_path, -3, -1)
            metadata = np.load(metadata_path)
            cameras = load_cameras(f'{DATA_ROOT}/hi4d/{seq_id}/cameras/rgb_cameras.npz')

            for cam_name, camera in cameras.items():
                pose_sampler = AdaptivePoseSampler2(
                    0.1, True, True, 100)
                for i_frame in range(int(metadata['start']), int(metadata['end']) + 1):
                    smpl_data = np.load(f'{DATA_ROOT}/hi4d/{seq_id}/smpl/{i_frame:06d}.npz')
                    pose = np.concatenate(
                        [smpl_data['global_orient'], smpl_data['body_pose']], axis=-1)
                    image_relpath = f'{seq_id}/images/{cam_name}/{i_frame:06d}.jpg'
                    mask_path = (f'{DATA_ROOT}/hi4d/{seq_id}/seg/img_seg_mask/{cam_name}/all/'
                                 f'{i_frame:06d}.png')
                    mask_all = np.any(imageio.imread(mask_path) != 0, axis=-1).astype(
                        np.uint8) * 255
                    masks_indiv = [imageio.imread(
                        f'{DATA_ROOT}/hi4d/{seq_id}/seg/img_seg_mask/{cam_name}/{i_person}/'
                        f'{i_frame:06d}.png') for i_person in range(metadata['num_persons'])]

                    encoded_masks = [RLEMask.from_array(mask) for mask in masks_indiv]
                    detections = detections_all[image_relpath]
                    dets_as_masks = [
                        RLEMask.from_bbox(b, imshape=mask_all.shape) for b in detections]
                    iou_matrix = RLEMask.iou_matrix(encoded_masks, dets_as_masks)


                    #encoded_masks = [rlemasklib.encode(mask) for mask in masks_indiv]
                    #detections = detections_all[image_relpath]
                    # dets_as_masks = [
                    #     rlemasklib.from_bbox(b, imshape=mask_all.shape) for b in detections]
                    # iou_matrix = np.array([[rlemasklib.iou([gt_mask, det_mask])
                    #                         for det_mask in dets_as_masks]
                    #                        for gt_mask in encoded_masks])
                    gt_indices, det_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)

                    for i_person, i_det in zip(gt_indices, det_indices):
                        if iou_matrix[i_person, i_det] < 0.1:
                            continue

                        joints = smpl_data['joints_3d'][i_person]
                        joints2d = camera.world_to_image(joints * 1000)
                        is_in_det = boxlib.contains(detections[i_det], joints2d)
                        if np.sum(is_in_det) < 6:
                            continue
                        # im = imageio.imread(f'{DATA_ROOT}/hi4d/{image_relpath}')
                        # drawing.draw_box(im, detections[i_det], color=(0, 255, 0))
                        # for i_joint, (x, y) in enumerate(joints2d):
                        #     color = (0, 0, 255) if is_in_det[i_joint] else (255, 0, 0)
                        #     drawing.circle(im, (x, y), radius=5, color=color)
                        # maskproc.draw_mask(im, masks_indiv[i_person], (0, 0, 255))

                        if pose_sampler.should_skip(joints):
                            continue
                        parameters = dict(
                            type='smpl', gender=metadata['genders'][i_person], pose=pose[i_person],
                            shape=smpl_data['betas'][i_person], trans=smpl_data['transl'][i_person],
                            kid_factor=np.float32(0))
                        bbox = detections[i_det][:4]
                        ex = ds3d.Pose3DExample(
                            image_path=f'hi4d/{image_relpath}',
                            camera=camera, bbox=bbox,
                            parameters=parameters, world_coords=None, mask=mask_all)
                        new_image_relpath = (
                            f'hi4d_downscaled/{seq_id}/images/{cam_name}/'
                            f'{i_frame:06d}_{i_person:02d}.jpg')

                        pool.apply_async(
                            make_efficient_example, (ex, new_image_relpath),
                            kwargs=dict(assume_image_ok=True),
                            callback=examples.append)

    ds = ds3d.Pose3DDataset(
        ds3d.JointInfo([], []),
        examples)

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/hi4d.barecat', readonly=False, overwrite=True,
            auto_codec=True) as bc:
        bc['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        for i, ex in enumerate(spu.progressbar(examples)):
            bc[f'train/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)

    return ds


def load_cameras(path):
    camdata = np.load(path)
    ids = camdata['ids']
    cameras = [cameravision.Camera(
        intrinsic_matrix=intr, rot_world_to_cam=extr[:, :3], trans_after_rot=extr[:, 3] * 1000,
        distortion_coeffs=dist, world_up=(0, 1, 0))
        for intr, extr, dist in zip(
            camdata['intrinsics'], camdata['extrinsics'], camdata['dist_coeffs'])]
    return dict(zip(ids, cameras))


if __name__ == '__main__':
    main()
