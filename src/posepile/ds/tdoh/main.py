import os.path as osp

import cameravision
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('tdoh.pkl', min_time="2021-12-12T22:25:46")
def make_dataset():
    root = f'{DATA_ROOT}/3doh'

    def make_examples(subdir, use_masks):
        annotations = spu.load_json(f'{root}/{subdir}/annots.json')
        image_dir = f'{root}/{subdir}/images'
        mask_dir = f'{root}/{subdir}/masks'
        examples = []
        with spu.ThrottledPool() as pool:
            for image_id, anno in spu.progressbar(annotations.items(), total=len(annotations)):
                image_path = f'{image_dir}/{image_id}.jpg'
                mask_path = f'{mask_dir}/{image_id}.png'
                camcoords = np.array(anno['lsp_joints_3d'], dtype=np.float32) * 1000 / 7

                extr = np.array(anno['extri'], np.float32)
                extr[:3, 3] *= 1000
                intr = np.array(anno['intri'], np.float32)
                world_up = -extr[1, :3]
                camera = cameravision.Camera(
                    extrinsic_matrix=extr, intrinsic_matrix=intr, world_up=world_up)

                (x1, y1), (x2, y2) = anno['bbox']
                bbox = np.array([x1, y1, x2 - x1, y2 - y1], np.float32)
                mask = imageio.imread(mask_path)[..., 0] if use_masks else None

                # Create example
                image_relpath = osp.relpath(image_path, DATA_ROOT)
                new_image_relpath = image_relpath.replace('3doh/', '3doh_downscaled/')
                world_coords = camera.camera_to_world(camcoords)
                ex = ds3d.Pose3DExample(image_relpath, world_coords, bbox, camera, mask=mask)
                pool.apply_async(
                    make_efficient_example, (ex, new_image_relpath), callback=examples.append)
        return examples

    examples_train = make_examples('trainset', use_masks=True)
    examples_test = make_examples('testset', use_masks=False)

    joint_names = 'rank,rkne,rhip,lhip,lkne,lank,rwri,relb,rsho,lsho,lelb,lwri,neck,head'
    edges = 'rsho-relb-rwri,rhip-rkne-rank,neck-head'
    joint_info = JointInfo(joint_names, edges)
    return ds3d.Pose3DDataset(joint_info, examples_train, [], examples_test)


if __name__ == '__main__':
    make_dataset()
