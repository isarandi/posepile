import argparse

import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.util.rigid_alignment as rigid_alignment
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-path', type=str, default=None)
    parser.add_argument('--procrustes', action=spu.argparse.BoolAction)
    spu.argparse.initialize(parser)
    all_true3d = get_all_gt_poses()
    all_pred3d = get_all_pred_poses(FLAGS.pred_path)
    if len(all_pred3d) != len(all_true3d):
        raise Exception(f'Unequal sample count! Pred: {len(all_pred3d)}, GT: {len(all_true3d)}')

    # Subtract root (we interpret it as the mean of left and right hip joints)
    all_pred3d -= np.mean(all_pred3d[:, 2:4], axis=1, keepdims=True)
    all_true3d -= np.mean(all_true3d[:, 2:4], axis=1, keepdims=True)

    if FLAGS.procrustes:
        all_pred3d = rigid_alignment.rigid_align_many(all_pred3d, all_true3d, scale_align=True)
    dist = np.linalg.norm(all_true3d - all_pred3d, axis=-1)
    overall_mean_error = np.mean(dist)
    print(overall_mean_error)


def get_all_gt_poses():
    annotations = spu.load_json(f'{DATA_ROOT}/3doh/testset/annots.json')
    image_ids = list(annotations.keys())
    coords_gt_cam = np.array([
        np.array(anno['lsp_joints_3d'], dtype=np.float32) * 1000 / 7
        for anno in annotations.values()])
    order = np.argsort(image_ids)
    return coords_gt_cam[order]


def get_all_pred_poses(path):
    results = np.load(path, allow_pickle=True)
    order = np.argsort(results['image_id'])
    return results['coords3d_pred_cam'][order]


if __name__ == '__main__':
    main()
