import functools
import importlib
import itertools

import cameravision
import cv2
import numpy as np
import rlemasklib
import simplepyutils as spu
from posepile import joint_filtering, util
from barecat.threadsafe import get_cached_reader
from posepile.joint_info import JointInfo
from posepile.util import TEST, TRAIN, VALID, improc


class Pose2DDataset:
    def __init__(
            self, joint_info, train_examples=None, valid_examples=None, test_examples=None):
        self.joint_info = joint_info
        self.examples = {
            TRAIN: train_examples or [], VALID: valid_examples or [], TEST: test_examples or []}

    def iter_examples(self):
        return itertools.chain(self.examples[TRAIN], self.examples[VALID], self.examples[TEST])


class Pose2DExample:
    def __init__(self, image_path, coords, bbox=None, mask=None, camera=None, densepose=None,
                 image=None):
        self.image_path = image_path
        self.coords = coords
        self.bbox = np.asarray(bbox)
        self.mask = mask
        self.camera = camera
        self.densepose = densepose
        self.image = image

    def get_image(self):
        if self.image is not None:
            return self.image
        return improc.imread(self.image_path)

    def load(self):
        return self


class Pose2DDatasetBarecat:
    def __init__(self, annotations_path, images_path):
        self.bc_annotations = get_cached_reader(annotations_path, auto_codec=True)
        self.bc_images = get_cached_reader(images_path, auto_codec=True)
        metadata = self.bc_annotations['metadata.msgpack']
        self.joint_info = JointInfo(metadata['joint_names'], metadata['joint_edges'])
        n_joints = self.joint_info.n_joints

        self.examples = {
            label: [
                Pose2DExampleBarecat(n_joints, annotations_path, images_path, p)
                for p in spu.progressbar(
                    self.bc_annotations.iglobfiles(f'{name}/**', recursive=True), desc=name
                )
            ]
            for label, name in zip([TRAIN, VALID, TEST], ['train', 'val', 'test'])
        }


class Pose2DExampleBarecat:
    __slots__ = ['bc_annotation_path', 'bc_image_path', 'path', 'n_joints']

    def __init__(self, n_joints, bc_annotation_path, bc_image_path, path_in_file):
        self.bc_annotation_path = bc_annotation_path
        self.bc_image_path = bc_image_path
        self.path = path_in_file
        self.n_joints = n_joints

    @property
    def image_path(self):
        return self.path.partition('/')[2].rpartition('_')[0]

    def load(self, load_image=True) -> Pose2DExample:
        d = get_cached_reader(self.bc_annotation_path)[self.path]
        ex = dict_to_example(d, self.n_joints)
        if load_image:
            try:
                ex.image = get_cached_reader(self.bc_image_path)[ex.image_path]
            except KeyError:
                ex.image = improc.imread(ex.image_path)
        return ex


@spu.picklecache('many2d.pkl', min_time="2020-02-01T02:53:21")
def make_many2d():
    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri,nose,leye,reye,lear,rear'
    edges = 'relb-rwri,rhip-rkne-rank'
    joint_info = JointInfo(joint_names, edges)
    datasets = [
        get_dataset('coco', single_person=False),
        get_dataset('mpii_yolo'),
        get_dataset('posetrack')]
    datasets = [joint_filtering.convert_dataset(ds, joint_info) for ds in datasets]

    body_joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,lelb,lwri'.split(',')
    body_joint_ids = [joint_info.ids[name] for name in body_joint_names]

    def n_valid_body_joints(example):
        return np.count_nonzero(np.all(~np.isnan(example.coords[body_joint_ids]), axis=-1))

    train_examples = [
        ex for ds in datasets
        for ex in ds.examples[TRAIN]
        if n_valid_body_joints(ex) > 6]
    return Pose2DDataset(joint_info, train_examples)


def make_huge2d():
    import posepile.merging.merged_dataset2d
    return posepile.merging.merged_dataset2d.make_huge2d()


def make_huge2d_common():
    import posepile.merging.merged_dataset2d
    return posepile.merging.merged_dataset2d.make_huge2d(separate=False)


def make_coco_reduced():
    import posepile.ds.coco.main
    return posepile.ds.coco.main.make_reduced_dataset()


def make_coco_wholebody():
    import posepile.ds.coco.main
    return posepile.ds.coco.main.make_wholebody()


def make_mpii_yolo():
    import posepile.ds.mpii.main
    return posepile.ds.mpii.main.make_mpii_yolo()


@functools.lru_cache()
def get_dataset(dataset_name, *args, **kwargs):
    from simplepyutils import FLAGS

    # Datasets can be loaded from pickle files
    if dataset_name.endswith('.pkl'):
        return spu.load_pickle(util.ensure_absolute_path(dataset_name))

    # Datasets can be loaded from barecat files
    if dataset_name.endswith('.barecat'):
        return Pose2DDatasetBarecat(dataset_name, FLAGS.image_barecat_path)

    try:
        make_fn = globals()[f'make_{dataset_name}']
    except KeyError:
        try:
            dataset_module = importlib.import_module(f'posepile.ds.{dataset_name}.main')
        except ImportError as e:
            raise ValueError(f'Dataset {dataset_name} not found.') from e

        make_fn = dataset_module.make_dataset

    return make_fn(*args, **kwargs)


def dict_to_example(d, n_joints):
    if 'cam' in d:
        cam = cameravision.Camera(
            rot_world_to_cam=cv2.Rodrigues(d['cam']['rotvec_w2c'])[0],
            optical_center=np.array(d['cam']['loc']),
            intrinsic_matrix=np.concatenate([d['cam']['intr'], np.array([[0, 0, 1]])]),
            distortion_coeffs=d['cam'].get('distcoef', None),
            world_up=d['cam'].get('up', (0, 0, 1))
        )
    else:
        cam = None

    coords = np.full(shape=[n_joints, 2], dtype=np.float32, fill_value=np.nan)
    j = d['joints2d']
    if len(j['i_rows']) > 0:
        coords[j['i_rows']] = j['rows']
    else:
        coords = j['rows']

    if 'densepose' in d:
        dp = d['densepose']
        densepose = (dp['imcoords'], dp['i_faces'], dp['barycoords'])
    else:
        densepose = None

    return Pose2DExample(
        image_path=d['impath'],
        bbox=d['bbox'].astype(np.float32),
        camera=cam,
        coords=coords,
        mask=rlemasklib.decompress(d['mask']) if 'mask' in d else None,
        densepose=densepose
    )
