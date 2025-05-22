import argparse
import glob
import os
import os.path as osp
import xml.etree.ElementTree

import boxlib
import cameravision
import numpy as np
import posepile.datasets3d as ds3d
import posepile.util.improc as improc
import simplepyutils as spu
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)
    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()
    elif FLAGS.stage == 3:
        make_rich_body()


def fix_detection_key(k):
    if k.startswith('./'):
        return k[2:]
    else:
        return k


def make_stage1():
    root = f'{DATA_ROOT}/rich'

    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    output_path = f'{DATA_ROOT}/rich_downscaled/stage1/stage1_{task_id:06d}.pkl'
    if osp.exists(output_path):
        return

    model_male, model_female = load_smplx_models()

    all_detections = spu.load_pickle(f'{root}/yolov4_detections.pkl')
    all_detections = {fix_detection_key(k): [np.array(x[:4]) for x in v]
                      for k, v in all_detections.items()}

    males = [0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 20]
    females = [13, 16, 18]

    train_seq_dirs = glob.glob(f'{root}/train/*')
    val_seq_dirs = glob.glob(f'{root}/val/*')
    im_seq_dirs = [*train_seq_dirs, *val_seq_dirs]
    print(len(im_seq_dirs))
    im_seq_dir = im_seq_dirs[task_id]

    examples = []
    with spu.ThrottledPool() as pool:
        seq_name = im_seq_dir.split('/')[-1]
        phase_name = im_seq_dir.split('/')[-2]
        setup = seq_name.split('_')[0]
        cam_ids = [int(n.split('_')[1]) for n in os.listdir(im_seq_dir)]
        i_frames = [
            int(n.split('_')[0]) for n in os.listdir(f'{im_seq_dir}/cam_{cam_ids[0]:02d}')]
        cameras = [
            load_camera(f'{root}/scan_calibration/{setup}/calibration/{cam_id:03d}.xml')
            for cam_id in cam_ids]

        subject_ids = [int(x) for x in seq_name.split('_')[1:-1]]
        unknown_genders = [s for s in subject_ids if s not in males and s not in females]
        if unknown_genders:
            raise Exception(f'Unknown genders for: {str(unknown_genders)}')

        body_models = {
            subject_id: (model_male if subject_id in males else model_female)
            for subject_id in subject_ids}
        pose_samplers = {
            i_subject: AdaptivePoseSampler2(100, True, True, 100)
            for i_subject in subject_ids}

        for i_frame in sorted(i_frames):
            fits = {subj: get_fit(
                f'{root}/{phase_name}_body/{seq_name}/{i_frame:05d}/{subj:03d}.pkl')
                for subj in subject_ids}
            fits = {subj: fit for subj, fit in fits.items() if fit is not None}
            world_poses = {
                subj: get_smplx_joints(body_models[subj], fit) for subj, fit in fits.items()}
            should_skips = {
                subj: pose_samplers[subj].should_skip(c) for subj, c in world_poses.items()}

            if all(should_skips.values()):
                continue

            for cam_id, camera in zip(cam_ids, cameras):
                if camera is None:
                    # the moving cameras have no calibration
                    continue

                image_path = f'{im_seq_dir}/cam_{cam_id:02d}/{i_frame:05d}_{cam_id:02d}.jpg'
                image_relpath = osp.relpath(image_path, root)
                imsize = improc.image_extents(image_path)

                for subj in world_poses:
                    if should_skips[subj]:
                        continue

                    world_coords = world_poses[subj]
                    imcoords = camera.world_to_image(world_coords)

                    bbox = get_bbox(imcoords, image_relpath, all_detections, imsize)
                    if np.min(bbox[2:]) < 100:
                        continue

                    ex = ds3d.Pose3DExample(
                        f'rich/{image_relpath}', world_coords, bbox=bbox, camera=camera)
                    image_relpath_noext = osp.splitext(image_relpath)[0]
                    new_image_relpath = f'rich_downscaled/{image_relpath_noext}_{subj:03d}.jpg'

                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        dict(downscale_input_for_antialias=True),
                        callback=examples.append)

    examples.sort(key=lambda ex: ex.image_path)
    spu.dump_pickle(examples, output_path)


# Stage2: generate the final dataset by incorporating the results of segmentation and preproc
@spu.picklecache('rich.pkl', min_time="2022-07-28T12:25:27")
def make_dataset():
    example_paths = glob.glob(f'{DATA_ROOT}/rich_downscaled/stage1/*.pkl')
    examples_all = []
    for p in example_paths:
        examples_all.extend(spu.load_pickle(p))

    examples_train = [ex for ex in examples_all if '/train/' in ex.image_path]
    examples_val = [ex for ex in examples_all if '/val/' in ex.image_path]

    joint_info = make_joint_info()
    ds = ds3d.Pose3DDataset(
        joint_info, train_examples=examples_train, valid_examples=examples_val)

    mask_paths = glob.glob(f'{DATA_ROOT}/rich_downscaled/masks/*.pkl')
    mask_dict = {}
    for path in mask_paths:
        mask_dict.update(spu.load_pickle(path))

    for ex in [*ds.examples[0], *ds.examples[1]]:
        relpath = spu.last_path_components(ex.image_path, 4)
        ex.mask = mask_dict[relpath]
    return ds


@spu.picklecache('rich_body.pkl', min_time="2022-08-05T17:49:02")
def make_rich_body():
    rich = make_dataset()
    # select body joints mostly
    i_sel = [*range(25), 28, 43, *range(55, 67), 68, 71, 73]
    for ex in [*rich.examples[0], *rich.examples[1], *rich.examples[2]]:
        ex.world_coords = np.ascontiguousarray(ex.world_coords[i_sel])
    rich.joint_info = rich.joint_info.select_joints(i_sel)
    return rich


def get_fit(path):
    try:
        return spu.load_pickle(path)
    except FileNotFoundError:
        return None


def load_smplx_models():
    import smplx
    model_folder = f'{DATA_ROOT}/body_models/smplx'
    model_male = smplx.create(
        model_folder, model_type='smplx', gender='male', use_face_contour=False, num_betas=10,
        num_pca_comps=12,
        num_expression_coeffs=10, use_pca=True, ext='npz')
    model_female = smplx.create(
        model_folder, model_type='smplx', gender='female', use_face_contour=False, num_betas=10,
        num_pca_comps=12,
        num_expression_coeffs=10, use_pca=True, ext='npz')
    return model_male, model_female


def load_camera(calib_path):
    if not osp.exists(calib_path):
        # the moving cameras have no calibration
        print(f'No {calib_path}')
        return None
    xml_root = xml.etree.ElementTree.parse(calib_path).getroot()
    extrinsic_matrix = np.array(xml_root.find('CameraMatrix').find('data').text.strip().split(),
                                np.float32).reshape(3, 4)
    extrinsic_matrix = np.concatenate([extrinsic_matrix, [[0, 0, 0, 1]]], axis=0)
    extrinsic_matrix[:3, 3] *= 1000
    intrinsic_matrix = np.array(
        xml_root.find('Intrinsics').find('data').text.strip().split(), np.float32).reshape(3, 3)
    dist_coeffs = np.array(
        xml_root.find('Distortion').find('data').text.strip().split(), np.float32)
    return cameravision.Camera(
        intrinsic_matrix=intrinsic_matrix, extrinsic_matrix=extrinsic_matrix,
        distortion_coeffs=dist_coeffs, world_up=-extrinsic_matrix[1, :3])


def make_joint_info():
    joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,'
        'lsho,rsho,lelb,relb,lwri,rwri,jaw,leyehf,reyehf,lindex1,lindex2,lindex3,lmiddle1,'
        'lmiddle2,lmiddle3,lpinky1,lpinky2,lpinky3,lring1,lring2,lring3,lthumb1,lthumb2,'
        'lthumb3,rindex1,rindex2,rindex3,rmiddle1,rmiddle2,rmiddle3,rpinky1,rpinky2,rpinky3,'
        'rring1,rring2,rring3,rthumb1,rthumb2,rthumb3,nose,reye,leye,rear,lear,lto2,lto3,lhee,'
        'rto2,rto3,rhee,lthu,lindex,lmiddle,lring,lpinky,rthu,rindex,rmiddle,rring,rpinky,'
        'reyebrow1,reyebrow2,reyebrow3,reyebrow4,reyebrow5,leyebrow5,leyebrow4,leyebrow3,'
        'leyebrow2,leyebrow1,nose1,nose2,nose3,nose4,rnose2,rnose1,nosemiddle,lnose1,lnose2,'
        'reye1,reye2,reye3,reye4,reye5,reye6,leye4,leye3,leye2,leye1,leye6,leye5,rmouth1,rmouth2,'
        'rmouth3,mouthtop,lmouth3,lmouth2,lmouth1,lmouth5,lmouth4,mouthbottom,rmouth4,rmouth5,'
        'rlip1,rlip2,toplip,llip2,llip1,llip3,bottomlip,rlip3')
    edges = (
        'rwri-relb-rsho-rcla-thor,rank-rhee,rto2-rtoe-rto3,'
        'rear-reye-nose-head-jaw-neck-thor-spin-bell-pelv-rhip-rkne-rank-rtoe')
    return JointInfo(joint_names, edges)


def get_bbox(im_coords, image_relpath, boxes, image_size):
    # TODO: first discard the points outside the image, then compute box
    # instead of first computing the box then intersecting it with the image
    bbox = boxlib.expand(boxlib.bb_of_points(im_coords), 1.05)

    if image_relpath in boxes and boxes[image_relpath]:
        candidates = boxes[image_relpath]
        ious = np.array([boxlib.iou(b, bbox) for b in candidates])
        i_best = np.argmax(ious)
        if ious[i_best] > 0.5:
            bbox = candidates[i_best]
    else:
        print(f'No detection {image_relpath}')

    return boxlib.intersection(bbox, boxlib.full(imsize=image_size))


def get_smplx_joints(body_model, fit):
    import torch
    body_torch = {k: torch.from_numpy(v) for k, v in fit.items()}
    output = body_model(**body_torch, return_verts=False)

    result = output.joints.detach().cpu().numpy()[0] * 1000
    return result.astype(np.float32)


if __name__ == '__main__':
    main()
