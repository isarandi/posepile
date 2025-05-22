import os.path as osp

import numpy as np
import posepile.datasets2d as ds2d
import src.posepile.util.drawing as drawing
import posepile.util.improc as improc
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example
import tarfile


@spu.picklecache('aic.pkl')
def make_dataset():
    joint_info = JointInfo(
        'rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,head,neck',
        'rwri-relb-rsho-neck,rank-rkne-rhip-lhip,neck-head,rsho-rhip')
    root = f'{DATA_ROOT}/aic'
    anno = spu.load_json(f'{root}/annotations/aic_train.json')
    filename_to_annos = spu.groupby(
        anno['annotations'], key=lambda a: anno['images'][a['image_id']]['file_name'])
    n_joints = 14

    UNLABELED = 0
    OCCLUDED = 1
    VISIBLE = 2

    examples = []
    with spu.ThrottledPool() as pool:
        with tarfile.open(f'{root}/AI_Challenger.tar.gz') as tarf:
            for member in spu.progressbar(tarf, total=len(anno['images'])):
                if not (member.isfile() and member.name.startswith('./train/images/')):
                    continue

                filename = osp.basename(member.name)
                annos_of_image = filename_to_annos.get(filename)
                if annos_of_image is None:
                    continue

                exs_of_image = []
                for i_person, ann in enumerate(annos_of_image):
                    bbox = ann['bbox']
                    bbox = np.array(
                        [bbox[0], bbox[1], bbox[2], bbox[3]], np.float32)
                    joints = np.array(ann['keypoints']).reshape([-1, 3])

                    visibilities = joints[:, 2]
                    coords = joints[:, :2].astype(np.float32).copy()
                    n_visible_joints = np.count_nonzero(visibilities == VISIBLE)
                    n_occluded_joints = np.count_nonzero(visibilities == OCCLUDED)
                    n_labeled_joints = n_occluded_joints + n_visible_joints

                    if n_visible_joints >= n_joints / 3 and n_labeled_joints >= n_joints / 2:
                        coords[visibilities == UNLABELED] = np.nan
                        ex = ds2d.Pose2DExample(image_path=None, coords=coords, bbox=bbox)
                        exs_of_image.append(ex)

                if exs_of_image:
                    im = improc.decode_jpeg_bytes(tarf.extractfile(member).read())
                    for i_person, ex in enumerate(exs_of_image):
                        # drawing.draw_box(im, ex.bbox, color=(0, 255, 0))
                        # for i_joint, (x, y) in enumerate(np.nan_to_num(ex.coords)):
                        #     drawing.circle(im, (x, y), radius=5, color=(255,0,0))
                        ex.image_path = im
                        filename_noext = osp.splitext(filename)[0]
                        new_im_path = (f'aic_downscaled/{filename_noext[:2]}/'
                                       f'{filename_noext[2:4]}/'
                                       f'{filename_noext}_{i_person:02d}.jpg')

                        pool.apply_async(
                            make_efficient_example, (ex, new_im_path), #kwargs=dict(assume_image_ok=True),
                            callback=examples.append)

    return ds2d.Pose2DDataset(joint_info, examples)


if __name__ == '__main__':
    make_dataset()
