import functools
import glob
import itertools
import os
import os.path as osp
import re
import xml.etree.ElementTree

import cameravision
import numpy as np
import simplepyutils as spu
import spacepy.pycdf
import transforms3d

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('h36m.pkl', min_time="2020-11-02T21:30:43")
def make_dataset(
        train_subjects=(1, 5, 6, 7, 8), valid_subjects=(), test_subjects=(9, 11),
        correct_S9=True, partial_visibility=False):
    joint_info = JointInfo(
        'rhip,rkne,rank,lhip,lkne,lank,tors,neck,head,htop,lsho,lelb,lwri,rsho,relb,rwri,pelv',
        'htop-head-neck-rsho-relb-rwri,neck-tors-pelv-rhip-rkne-rank')

    if not spu.all_disjoint(train_subjects, valid_subjects, test_subjects):
        raise Exception('Set of train, val and test subject must be disjoint.')

    # use last subject of the non-test subjects for validation
    train_examples = []
    test_examples = []
    valid_examples = []

    if partial_visibility:
        dir_suffix = '_partial'
        further_expansion_factor = 1.8
    else:
        dir_suffix = '' if correct_S9 else 'incorrect_S9'
        further_expansion_factor = 1

    with spu.ThrottledPool() as pool:
        for i_subject in [*test_subjects, *train_subjects, *valid_subjects]:
            if i_subject in train_subjects:
                examples_container = train_examples
            elif i_subject in valid_subjects:
                examples_container = valid_examples
            else:
                examples_container = test_examples

            frame_step = 5 if i_subject in train_subjects else 64

            for activity_name, camera_id in itertools.product(
                    get_activity_names(i_subject), range(4)):
                print(f'Processing S{i_subject} {activity_name} {camera_id}')
                image_relpaths, world_coords_all, bboxes, camera = get_examples(
                    i_subject, activity_name, camera_id, frame_step=frame_step,
                    correct_S9=correct_S9)
                pose_sampler = AdaptivePoseSampler(100)
                for image_relpath, world_coords, bbox in zip(
                        spu.progressbar(image_relpaths), world_coords_all, bboxes):
                    # Using very similar examples is wasteful when training. Therefore:
                    # skip frame if all keypoints are within a distance compared to last stored
                    # frame. This is not done when testing, as it would change the results.
                    if i_subject in train_subjects and pose_sampler.should_skip(world_coords):
                        continue
                    activity_name = activity_name.split(' ')[0]
                    ex = ds3d.Pose3DExample(
                        image_relpath, world_coords, bbox, camera, activity_name=activity_name)
                    new_image_relpath = image_relpath.replace(
                        'h36m', f'h36m_downscaled{dir_suffix}')
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath, further_expansion_factor),
                        callback=examples_container.append)

    train_examples.sort(key=lambda x: x.image_path)
    valid_examples.sort(key=lambda x: x.image_path)
    test_examples.sort(key=lambda x: x.image_path)
    return ds3d.Pose3DDataset(joint_info, train_examples, valid_examples, test_examples)


@spu.picklecache('h36m_masked.pkl', min_time="2020-11-02T21:30:43")
def make_h36m_masked():
    ds = make_dataset()
    mask_paths = glob.glob(f'{DATA_ROOT}/h36m_downscaled/masks/*.pkl')
    mask_dict = {}
    for path in mask_paths:
        mask_dict.update(spu.load_pickle(path))

    print(len(ds.examples[0]))
    for ex in ds.examples[0]:
        relpath = spu.last_path_components(ex.image_path, 4)
        ex.mask = mask_dict[relpath]
    return ds


def correct_boxes(bboxes, path, world_coords, camera):
    """Three activties for subject S9 have erroneous bounding boxes, they are horizontally shifted.
    This function corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""

    def correct_image_coords(bad_imcoords):
        root_depths = camera.world_to_camera(world_coords[:, -1])[:, 2:]
        bad_worldcoords = camera.image_to_world(bad_imcoords, camera_depth=root_depths)
        good_worldcoords = bad_worldcoords + np.array([-200, 0, 0])
        good_imcoords = camera.world_to_image(good_worldcoords)
        return good_imcoords

    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        toplefts = correct_image_coords(bboxes[:, :2])
        bottomrights = correct_image_coords(bboxes[:, :2] + bboxes[:, 2:])
        return np.concatenate([toplefts, bottomrights - toplefts], axis=-1)

    return bboxes


def correct_world_coords(coords, path):
    """Three activties for subject S9 have erroneous coords, they are horizontally shifted.
    This corrects them. Use --dataset=h36m-incorrect-S9 to use the erroneous annotation."""
    if 'S9' in path and ('SittingDown 1' in path or 'Waiting 1' in path or 'Greeting.' in path):
        coords = coords.copy()
        coords[:, :, 0] -= 200
    return coords


def get_examples(i_subject, activity_name, i_camera, frame_step=5, correct_S9=True):
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    camera_name = camera_names[i_camera]
    camera = get_cameras()[i_camera][i_subject - 1]

    i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    n_total_frames, world_coords = load_coords(
        i_subject, activity_name, i_relevant_joints, frame_step, correct_S9)

    image_relfolder = f'h36m/S{i_subject}/Images/{activity_name}.{camera_name}'
    image_relpaths = [f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                      for i_frame in range(0, n_total_frames, frame_step)]

    bbox_path = f'{DATA_ROOT}/h36m/S{i_subject}/BBoxes/{activity_name}.{camera_name}.npy'
    bboxes = np.load(bbox_path)[::frame_step]
    if correct_S9:
        bboxes = correct_boxes(bboxes, bbox_path, world_coords, camera)

    return image_relpaths, world_coords, bboxes, camera


@functools.lru_cache(20)
def load_coords(i_subject, activity_name, i_relevant_joints=None, frame_step=5, correct_S9=False):
    pose_folder = f'{DATA_ROOT}/h36m/S{i_subject}/MyPoseFeatures'
    coord_path = f'{pose_folder}/D3_Positions/{activity_name}.cdf'

    with spacepy.pycdf.CDF(coord_path) as cdf_file:
        coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
    coords_raw = coords_raw_all[::frame_step]
    coords_new_shape = [coords_raw.shape[0], -1, 3]
    n_frames = coords_raw_all.shape[0]
    world_coords = coords_raw.reshape(coords_new_shape)
    if i_relevant_joints is not None:
        world_coords = world_coords[:, i_relevant_joints]

    if correct_S9:
        world_coords = correct_world_coords(world_coords, coord_path)
    return n_frames, world_coords


@functools.lru_cache()
def get_cameras():
    metadata_path = f'{DATA_ROOT}/h36m/Release-v1.2/metadata.xml'
    root = xml.etree.ElementTree.parse(metadata_path).getroot()
    cam_params_text = root.findall('w0')[0].text
    numbers = np.array([float(x) for x in cam_params_text[1:-1].split(' ')])
    extrinsic = numbers[:264].reshape(4, 11, 6)
    intrinsic = numbers[264:].reshape(4, 9)

    cameras = [[make_h36m_camera(extrinsic[i_camera, i_subject], intrinsic[i_camera])
                for i_subject in range(11)]
               for i_camera in range(4)]
    return cameras


def make_h36m_camera(extrinsic_params, intrinsic_params):
    R = transforms3d.euler.euler2mat(*extrinsic_params[:3], 'rxyz')
    t = extrinsic_params[3:6]
    f, c, k, p = np.split(intrinsic_params, (2, 4, 7))
    distortion_coeffs = np.array([k[0], k[1], p[0], p[1], k[2]], np.float32)
    intrinsic_matrix = np.array([
        [f[0], 0, c[0]],
        [0, f[1], c[1]],
        [0, 0, 1]], np.float32)
    return cameravision.Camera(t, R, intrinsic_matrix, distortion_coeffs)


def get_activity_names(i_subject):
    h36m_root = f'{DATA_ROOT}/h36m/'
    subject_images_root = f'{h36m_root}/S{i_subject}/Images/'
    subdirs = [elem for elem in os.listdir(subject_images_root)
               if osp.isdir(f'{subject_images_root}/{elem}')]
    activity_names = set(elem.split('.')[0] for elem in subdirs if '_' not in elem)
    return sorted(activity_names)


@spu.picklecache('h36m_alljoints.pkl', min_time="2021-12-10T01:15:00")
def make_h36m_alljoints():
    i_relevant_joints = tuple(
        [*range(1, 11), 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30, 0])
    joint_names = (
        'rhip,rkne,rank,rfoo,rtoe,lhip,lkne,lank,lfoo,ltoe,spin,neck,head,htop,'
        'lsho,lelb,lwri,lthu,lfin,rsho,relb,rwri,rthu,rfin,pelv')
    edges = ('htop-head-neck-spin-pelv-lhip-lkne-lank-lfoo-ltoe,'
             'lthu-lwri-lelb-lsho-neck-rsho-relb-rwri-rthu,rwri-rfin,lwri-lfin,'
             'pelv-rhip-rkne-rank-rfoo-rtoe')
    joint_info = JointInfo(joint_names, edges)
    h36m = make_h36m_masked()
    for ex in spu.progressbar([*h36m.examples[0], *h36m.examples[1], *h36m.examples[2]]):
        image_path = ex.image_path
        pattern = (r'.+/S(?P<i_subject>\d+)/Images/(?P<action>.+)\.(?P<cam>\d+)/frame_('
                   r'?P<i_frame>\d+).jpg')
        m = re.match(pattern, image_path)
        n_frames, coords = load_coords(m['i_subject'], m['action'], i_relevant_joints, frame_step=1)
        i_frame = int(m['i_frame'])
        coords = coords[i_frame]
        ex.world_coords = coords
        ex.univ_coords = None
    h36m.joint_info = joint_info
    return h36m


def get_all_gt_poses(i_subjects, i_relevant_joints, frame_step):
    camera_names = ['54138969', '55011271', '58860488', '60457274']
    all_world_coords = []
    all_image_relpaths = []
    for i_subj in i_subjects:
        for activity, cam_id in itertools.product(get_activity_names(i_subj), range(4)):
            # Corrupt data in original release?
            # Maybe not with every video codec? Seems to work now.
            # if i_subj == 11 and activity == 'Directions' and cam_id == 0:
            #    continue
            n_frames_total, world_coords = load_coords(
                i_subject=i_subj, activity_name=activity,
                i_relevant_joints=tuple(i_relevant_joints),
                frame_step=frame_step)
            all_world_coords.append(world_coords)
            camera_name = camera_names[cam_id]
            image_relfolder = f'h36m/S{i_subj}/Images/{activity}.{camera_name}'
            all_image_relpaths += [
                f'{image_relfolder}/frame_{i_frame:06d}.jpg'
                for i_frame in range(0, n_frames_total, frame_step)]

    order = np.argsort(all_image_relpaths)
    all_world_coords = np.concatenate(all_world_coords, axis=0)[order]
    all_image_relpaths = np.array(all_image_relpaths)[order]
    return all_image_relpaths, all_world_coords


if __name__ == '__main__':
    make_dataset()
