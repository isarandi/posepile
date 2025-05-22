import argparse
import functools
import glob
import io
import os
import os.path as osp

import PIL.Image
import cameralib
import cv2
import numpy as np
import simplepyutils as spu
import tensorflow as tf
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)
    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


def make_stage1():
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    output_path = f'{DATA_ROOT}/hspace_downscaled/stage1/stage1_{task_id:06d}.pkl'
    if osp.exists(output_path):
        return

    root = f'{DATA_ROOT}/hspace'
    train_paths_all = sorted(tf.io.gfile.glob(f'{root}/2021_01/train/*.tfrecord'))
    train_paths_task = train_paths_all[task_id * 20:(task_id + 1) * 20]

    i_sel, joint_info = make_joint_info()

    examples = []
    with spu.ThrottledPool() as pool:
        for tfrecord_filepath in spu.progressbar(train_paths_task):
            sampler = AdaptivePoseSampler2(100, True, True, 20)
            print(tfrecord_filepath)
            tf_dataset = tf.data.TFRecordDataset(tfrecord_filepath)
            record_name = osp.splitext(osp.basename(tfrecord_filepath))[0]

            for i_example, serialized_example in enumerate(tf_dataset.as_numpy_iterator()):
                tf_example = tf.train.Example.FromString(serialized_example)
                n_people = get_num_people(tf_example)
                pose_per_person = [
                    get_person_joints(tf_example, i_person)[i_sel]
                    for i_person in range(n_people)]

                should_skips = [sampler.should_skip(pose) for pose in pose_per_person]
                if all(should_skips):
                    continue

                image = get_image(tf_example)
                labels, overall_mask = get_mask(tf_example)
                camera = get_camera(tf_example)

                for i_person, (world_coords, should_skip) in enumerate(
                        zip(pose_per_person, should_skips)):
                    if should_skip:
                        continue

                    person_mask = get_person_mask(tf_example, labels, i_person)
                    mask_box = np.array(cv2.boundingRect(person_mask), np.float32)
                    # Skip tiny examples
                    if min(mask_box[2:4]) < 100:
                        continue

                    ex = ds3d.Pose3DExample(
                        image, world_coords, mask_box, camera, mask=overall_mask)
                    new_image_relpath = (
                        f'hspace_downscaled/train/{record_name}/'
                        f'{i_example:08d}_{i_person:03d}.jpg')
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        callback=examples.append)

    spu.dump_pickle(examples, output_path)


@spu.picklecache('hspace.pkl', min_time="2021-12-09T21:09:38")
def make_dataset():
    example_paths = glob.glob(f'{DATA_ROOT}/hspace_downscaled/stage1/*.pkl')
    examples_all = []
    for p in example_paths:
        examples_all.extend(spu.load_pickle(p))

    i_sel, joint_info = make_joint_info()
    return ds3d.Pose3DDataset(joint_info, train_examples=examples_all)


def get_camera(tf_example):
    fx, fy, cx, cy = tf.constant(
        tf_example.features.feature['camera_intrinsics'].float_list.value,
        tf.float32).numpy()
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    return cameralib.Camera(intrinsic_matrix=intrinsic_matrix, world_up=(0, -1, 0))


def get_person_joints(tf_example, i_person):
    values = tf_example.features.feature[f'RIGGED_{i_person}/joints_3d'].float_list.value
    return tf.reshape(values, (-1, 3)).numpy() * 1000


@functools.lru_cache
def get_structuring_element():
    return cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))


def get_mask(tf_example):
    mask_bytes = tf_example.features.feature['character_masks'].bytes_list.value[0]
    labels = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    overall_mask = (labels > 0).astype(np.uint8) * 255
    process_mask_morphological(overall_mask)
    return labels, overall_mask


def process_mask_morphological(mask):
    element = get_structuring_element()
    cv2.erode(mask, element, dst=mask, iterations=1)
    cv2.dilate(mask, element, dst=mask, iterations=2)


def get_image(tf_example):
    value = tf_example.features.feature['image/rgb'].bytes_list.value[0]
    return np.array(PIL.Image.open(io.BytesIO(value)))[..., ::-1]


def get_num_people(tf_example):
    return tf_example.features.feature['number_of_rigged_characters'].int64_list.value[0]


def get_person_mask(tf_example, labels, i_person):
    label = tf_example.features.feature[f'RIGGED_{i_person}/mask_id'].int64_list.value[0]
    mask = ((labels == label) | (labels == (label - 1))).astype(np.uint8) * 255
    process_mask_morphological(mask)
    return mask


def make_joint_info():
    joint_names = (
        'pelv,spin1,spin2,spin3,neck,head,head2,lbrow1,lbrow2,rbrow1,rbrow2,'
        'lbrow3,rbrow3,lcla,lsho,lelb,lwri,lindex1,lindex2,lindex3,lmiddle1,'
        'lmiddle2,lmiddle3,lring1,lring2,lring3,lpinkie1,lpinkie2,lpinkie3,lthu1,'
        'lthu2,lthu,rcla,rsho,relb,rwri,rindex1,rindex2,rindex3,rmiddle1,'
        'rmiddle2,rmiddle3,rring1,rring2,rring3,rpinkie1,rpinkie2,rpinkie3,rthu1,'
        'rthu2,rthu,lhip,lkne,lank,lhee,lfoo,ltoe,rhip,rkne,rank,rhee,rfoo,rtoe')
    edges = (
        'rthu-rwri-relb-rsho-rcla-spin3,'
        'leye1-nose-head-neck-spin3-spin2-spin1-pelv-rhip-rkne-rank-rhee-rfoo-rtoe')
    joint_info = JointInfo(joint_names, edges)
    selection = ('pelv,spin1,spin2,spin3,neck,head,head2,lbrow1,rbrow1,lcla,lsho,'
                 'lelb,lwri,lmiddle1,lmiddle3,lthu,rcla,rsho,relb,rwri,rmiddle1,'
                 'rmiddle3,rthu,lhip,lkne,lank,lhee,lfoo,ltoe,rhip,rkne,rank,rhee,'
                 'rfoo,rtoe').split(',')
    i_sel = [joint_info.ids[name] for name in selection]
    joint_info = joint_info.select_joints(i_sel)
    return i_sel, joint_info


if __name__ == '__main__':
    main()
