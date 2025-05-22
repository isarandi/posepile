import os.path as osp

import boxlib
import cv2
import numpy as np
import posepile.datasets2d as ds2d
import src.posepile.util.drawing as drawing
import posepile.util.improc as improc
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('halpe2.pkl', min_time="2024-02-08T16:57:32")
def make_dataset():
    joint_info = get_joint_info()
    root = f'{DATA_ROOT}/halpe'
    anno = spu.load_json(f'{root}/halpe_train_v1.json')

    image_id_to_filename = {i['id']: i['file_name'] for i in anno['images']}

    def get_image_name_of_anno(a):
        image_id = a['image_id']
        try:
            return image_id_to_filename[image_id]
        except IndexError:
            print(f'Not found: {image_id}')
            return ''

    filename_to_annos = spu.groupby(anno['annotations'], key=get_image_name_of_anno)
    n_body_joints = 26

    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2

    examples = []
    with spu.ThrottledPool() as pool:
        for filename, annos_of_image in spu.progressbar_items(filename_to_annos):
            if filename == '':
                continue

            exs_of_image = {}
            for i_person, ann in enumerate(annos_of_image):
                bbox = ann['bbox']
                bbox = np.array(
                    [bbox[0], bbox[1], bbox[2], bbox[3]], np.float32)
                joints = np.array(ann['keypoints']).reshape([-1, 3])

                visibilities = joints[:, 2]
                coords = joints[:, :2].astype(np.float32).copy()
                n_visible_bodyjoints = np.count_nonzero(visibilities[:26] == VISIBLE)
                n_occluded_bodyjoints = np.count_nonzero(visibilities[:26] == OCCLUDED)
                n_labeled_bodyjoints = n_visible_bodyjoints + n_occluded_bodyjoints
                coords[visibilities == UNLABELED] = np.nan

                is_zero = np.all(coords ==0, axis=1)

                # coords[is_zero] = np.nan  # SHOULD ACTIVATE THIS

                if (n_visible_bodyjoints >= n_body_joints / 3 and
                        n_labeled_bodyjoints >= n_body_joints / 2 and np.min(bbox[2:]) > 100 and
                        np.min(boxlib.bb_of_points(coords)[2:]) > 100):

                    exs_of_image[i_person] = ds2d.Pose2DExample(
                        image_path=None, coords=coords, bbox=bbox)

            if exs_of_image:
                try:
                    im = improc.imread(f'{root}/hico_20160224_det/images/train2015/{filename}')
                except Exception as e:
                    print(f'Error reading {filename}')
                    print(e)
                    print('Continuing...')
                    continue
                for i_person, ex in exs_of_image.items():
                    #im2 = im.copy()
                    #drawing.draw_box(im2, ex.bbox, color=(0, 255, 0))
                    #draw_stick_figure_2d_inplace(
                    #   im2, ex.coords, joint_info.stick_figure_edges, 2, (0, 255, 0))
                    ex.image_path = im
                    filename_noext = osp.splitext(filename)[0]
                    new_im_path = (f'halpe_downscaled/{filename_noext[-2:]}/'
                                   f'{filename_noext}_{i_person:02d}.jpg')

                    pool.apply_async(
                        make_efficient_example, (ex, new_im_path),
                        # kwargs=dict(assume_image_ok=True),
                        callback=examples.append)

    return ds2d.Pose2DDataset(joint_info, examples)


def get_joint_info():
    names = ('nose,leye,reye,lear,rear,lsho,rsho,lelb,relb,lwri,rwri,lhip,rhip,lkne,rkne,lank,'
             'rank,htop,neck,pelv,ltoe,rtoe,ltoe2,rtoe2,lhee,rhee,rface1,rface2,rface3,rface4,'
             'rface5,rface6,rface7,rface8,chin,lface8,lface7,lface6,lface5,lface4,lface3,lface2,'
             'lface1,rface9,rface10,rface11,rface12,rface13,lface13,lface12,lface11,lface10,'
             'lface9,nose1,nose2,nose3,nose4,rface14,rface15,nose5,lface15,lface14,reye1,reye2,'
             'reye3,reye4,reye5,reye6,leye4,leye3,leye2,leye1,leye6,leye5,rmouth1,rmouth2,'
             'rmouth3,mouth1,lmouth3,lmouth2,lmouth1,lmouth4,lmouth5,mouth2,rmouth5,rmouth4,'
             'rmouth6,rmouth7,mouth3,lmouth7,lmouth6,lmouth8,mouth4,rmouth8,lhan1,lhan2,lhan3,'
             'lhan4,lhan5,lhan6,lhan7,lhan8,lhan9,lhan10,lhan11,lhan12,lhan13,lhan14,lhan15,'
             'lhan16,lhan17,lhan18,lhan19,lhan20,lhan21,rhan1,rhan2,rhan3,rhan4,rhan5,rhan6,'
             'rhan7,rhan8,rhan9,rhan10,rhan11,rhan12,rhan13,rhan14,rhan15,rhan16,rhan17,rhan18,'
             'rhan19,rhan20,rhan21')
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
        (17, 18), (18, 19), (19, 11), (19, 12),
        (11, 13), (12, 14), (13, 15), (14, 16),
        (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
        (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
        (35, 36), (36, 37), (37, 38),
        # Face
        (38, 39), (39, 40), (40, 41), (41, 42), (43, 44), (44, 45), (45, 46), (46, 47), (48, 49),
        (49, 50), (50, 51), (51, 52),
        # Face
        (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61), (62, 63), (63, 64),
        (64, 65), (65, 66), (66, 67),
        # Face
        (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (74, 75), (75, 76), (76, 77), (77, 78),
        (78, 79), (79, 80), (80, 81),
        # Face
        (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90),
        (90, 91), (91, 92), (92, 93),
        # Face
        (94, 95), (95, 96), (96, 97), (97, 98), (94, 99), (99, 100), (100, 101), (101, 102),
        (94, 103), (103, 104), (104, 105),
        # LeftHand
        (105, 106), (94, 107), (107, 108), (108, 109), (109, 110), (94, 111), (111, 112),
        (112, 113), (113, 114),
        # LeftHand
        (115, 116), (116, 117), (117, 118), (118, 119), (115, 120), (120, 121), (121, 122),
        (122, 123), (115, 124), (124, 125),
        # RightHand
        (125, 126), (126, 127), (115, 128), (128, 129), (129, 130), (130, 131), (115, 132),
        (132, 133), (133, 134), (134, 135)
        # RightHand
    ]
    return JointInfo(names, edges)


def draw_stick_figure_2d_inplace(im, coords, joint_edges, thickness=3, color=None):
    for i_joint1, i_joint2 in joint_edges:
        relevant_coords = coords[[i_joint1, i_joint2]]
        if not np.isnan(relevant_coords).any() and not np.isclose(0, relevant_coords).any():
            drawing.line(
                im, coords[i_joint1], coords[i_joint2],
                color=color, thickness=thickness, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    make_dataset()
