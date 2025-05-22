import collections
import os.path as osp

import boxlib
import cameravision
import numpy as np
import posepile.datasets3d as ds3d
import posepile.util.improc as improc
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('gpa.pkl', min_time="2021-12-05T21:47:55")
def make_dataset():
    gpa_root = f'{DATA_ROOT}/gpa'
    image_dir = f'{gpa_root}/Gaussian_img_jpg_new'
    mask_dir = f'{gpa_root}/img_jpg_new_resnet101deeplabv3humanmask'
    annot = spu.load_json(f'{gpa_root}/xyz_gpa12_mdp_cntind_crop_cam_c2g.json')
    all_detections = spu.load_pickle(f'{gpa_root}/yolov4_detections.pkl')
    cam_ids = np.load(f'{gpa_root}/Sequence_ids/cam_ids_750k.npy')
    examples_train = []
    examples_test = []
    with spu.ThrottledPool() as pool:
        pose_sample_per_cam = collections.defaultdict(lambda: AdaptivePoseSampler(100))
        for image_info, anno in zip(spu.progressbar(annot['images']), annot['annotations']):
            image_id = image_info['file_name'][-14:-4]
            image_path = f'{image_dir}/{image_id[:6]}/{image_id[6:8]}/{image_id}.jpg'

            # If no detection, then skip
            detections = all_detections[osp.relpath(image_path, image_dir)]
            if detections.size == 0:
                continue
            camcoords = np.array(anno['joint_cams'], dtype=np.float32).T

            # Filter for movement
            if pose_sample_per_cam[cam_ids[int(image_id)]].should_skip(camcoords):
                continue

            # Check which detection matches best
            camera = get_camera(anno)
            imcoords = camera.camera_to_image(camcoords)
            posebox = boxlib.expand(boxlib.bb_of_points(imcoords), 1.15)
            ious = [boxlib.iou(posebox, d) for d in detections]
            if np.max(ious) < 0.3:
                continue

            i_best_det = np.argmax(ious)
            box = detections[i_best_det]

            # Create example
            image_relpath = osp.relpath(image_path, DATA_ROOT)
            new_image_relpath = image_relpath.replace('gpa/', 'gpa_downscaled/')
            mask_path = f'{mask_dir}/{image_id[:6]}/{image_id[6:8]}/{image_id}_mask.jpg'
            mask = (improc.imread(mask_path)[..., 0] > 10).astype(np.uint8) * 255
            ex = ds3d.Pose3DExample(image_relpath, camcoords, box, camera, mask=mask)
            pool.apply_async(
                make_efficient_example, (ex, new_image_relpath),
                callback=(examples_train if anno['istrains'] else examples_test).append)

    joint_names = (
        'pelv,spin,spi1,spi2,spi3,neck,head,htop,rcla,rsho,relb,rwri,rhan,rfin,rthu,rthu2,lcla,'
        'lsho,lelb,lwri,lhan,lfin,lthu,lthu2,rhip,rkne,rank,rfoo,rtoe,lhip,lkne,lank,lfoo,ltoe')
    edges = (
        'rtoe-rfoo-rank-rkne-rhip-pelv-spin-spi1-spi2-spi3-neck-head-htop,'
        'spi3-rcla-rsho-relb-rwri-rhan-rfin,rwri-rthu-rthu2')
    joint_info = JointInfo(joint_names, edges)
    return ds3d.Pose3DDataset(joint_info, examples_train, [], examples_test)


def get_camera(anno):
    intr = np.array([
        [anno['f'][0], 0, anno['c'][0]],
        [0, anno['f'][1], anno['c'][1]],
        [0, 0, 1]], dtype=np.float32)
    camera = cameravision.Camera(intrinsic_matrix=intr, world_up=(0, -1, 0))
    return camera


if __name__ == '__main__':
    make_dataset()
