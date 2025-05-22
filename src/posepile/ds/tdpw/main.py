import glob
import os.path as osp

import boxlib
import cameravision
import numpy as np
import simplepyutils as spu

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT
from posepile.util.preproc_for_efficiency import make_efficient_example


@spu.picklecache('tdpw.pkl', min_time="2021-07-09T12:26:16")
def make_dataset():
    root = f'{DATA_ROOT}/3dpw'
    body_joint_names = (
        'pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,rcla,head,lsho,'
        'rsho,lelb,relb,lwri,rwri,lhan,rhan'.split(','))
    selected_joints = [*range(1, 24), 0]
    joint_names = [body_joint_names[j] for j in selected_joints]
    edges = 'head-neck-thor-rcla-rsho-relb-rwri-rhan,thor-spin-bell-pelv-rhip-rkne-rank-rtoe'
    joint_info = JointInfo(joint_names, edges)

    def get_examples(phase, pool):
        result = []
        seq_filepaths = glob.glob(f'{root}/sequenceFiles/{phase}/*.pkl')
        for filepath in seq_filepaths:
            seq = spu.load_pickle(filepath)
            seq_name = seq['sequence']
            intrinsics = seq['cam_intrinsics']
            extrinsics_per_frame = seq['cam_poses']

            for i_person, (coord_seq, coords2d_seq, trans_seq, camvalid_seq) in enumerate(zip(
                    seq['jointPositions'], seq['poses2d'], seq['trans'], seq['campose_valid'])):
                for i_frame, (coords, coords2d, trans, extrinsics, campose_valid) in enumerate(
                        zip(coord_seq, coords2d_seq, trans_seq, extrinsics_per_frame,
                            camvalid_seq)):
                    if not campose_valid or np.all(coords2d == 0):
                        continue

                    impath = f'{root}/imageFiles/{seq_name}/image_{i_frame:05d}.jpg'

                    camera = cameravision.Camera(
                        extrinsic_matrix=extrinsics, intrinsic_matrix=intrinsics,
                        world_up=(0, 1, 0))
                    camera.t *= 1000
                    world_coords = (coords.reshape(-1, 3))[selected_joints] * 1000
                    camera2 = cameravision.Camera(intrinsic_matrix=intrinsics, world_up=(0, -1, 0))
                    camcoords = camera.world_to_camera(world_coords)
                    imcoords = camera.world_to_image(world_coords)
                    bbox = boxlib.expand(boxlib.bb_of_points(imcoords), 1.15)
                    ex = ds3d.Pose3DExample(impath, camcoords, bbox=bbox, camera=camera2)
                    noext, ext = osp.splitext(osp.relpath(impath, root))
                    new_image_relpath = f'tdpw_downscaled/{noext}_{i_person:03d}.jpg'
                    pool.apply_async(
                        make_efficient_example,
                        (ex, new_image_relpath, 1, False, "2021-07-09T12:28:07"),
                        callback=result.append)
        return result

    with spu.ThrottledPool() as pool:
        train_examples = get_examples('train', pool)
        val_examples = get_examples('validation', pool)
        test_examples = get_examples('test', pool)

    # Use all for testing
    test_examples = [*train_examples, *val_examples, *test_examples]
    test_examples.sort(key=lambda ex: ex.image_path)
    return ds3d.Pose3DDataset(joint_info, test_examples=test_examples)


if __name__ == '__main__':
    make_dataset()
