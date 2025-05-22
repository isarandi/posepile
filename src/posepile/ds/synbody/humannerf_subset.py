import zipfile

import boxlib
import cameralib
import imageio.v2 as imageio
import numpy as np
import posepile.datasets3d as ds3d
import simplepyutils as spu
import smpl.numpy
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example

DATASET_NAME = 'synbody'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}/HumanNeRF-subset'


@spu.picklecache(f'{DATASET_NAME}_humannerf.pkl', min_time="2023-12-20T02:00:11")
def make_dataset():
    examples = []
    zipf = zipfile.ZipFile(f'{DATASET_DIR}/img.zip')

    with spu.ThrottledPool() as pool:
        for i_seq in spu.progressbar(range(100)):
            cameras = load_cameras(f'{DATASET_DIR}/{i_seq:03d}/camera.json')
            smpl_refit = np.load(
                f'{DATASET_DIR}/{i_seq:03d}/smpl_refit.npz', allow_pickle=True)
            gender = smpl_refit['meta'].item()['gender']
            smpl_data = smpl_refit['smpl'].item()
            pose_seq = np.concatenate(
                [smpl_data['global_orient'], smpl_data['body_pose']], axis=1)
            shape_seq = smpl_data['betas']
            trans_seq = smpl_data['transl']

            sampler = AdaptivePoseSampler2(
                100, True, True, 100)
            for i_frame, (pose, shape, trans) in enumerate(
                    zip(spu.progressbar(pose_seq, desc=f'{i_seq:03d}', leave=False),
                        shape_seq, trans_seq)):
                parameters = dict(
                    type='smpl', gender=gender, pose=pose.reshape(-1), shape=shape, trans=trans)

                joints = smpl.numpy.get_cached_body_model(
                    'smpl', gender).single(pose, shape, trans)['joints'] * 1000
                if sampler.should_skip(joints):
                    continue

                for i_cam, camera in enumerate(cameras):
                    try:
                        mask = imageio.imread(
                            f'{DATASET_DIR}/{i_seq:03d}/mask/{i_cam:02d}/{i_frame:04d}.png')
                    except FileNotFoundError:
                        print(f'Missing mask for {i_seq:03d}/{i_cam:02d}/{i_frame:04d}.png')
                        continue

                    impath = f'{i_seq:03d}/img/{i_cam:02d}/{i_frame:04d}.jpg'
                    im = np.frombuffer(zipf.open(impath).read(), np.uint8)
                    bbox = boxlib.bb_of_mask(mask)

                    # verts2d = camera.world_to_image(verts*1000)
                    # for x,y in verts2d:
                    #    cv2.circle(im, (int(x),int(y)), 3, (255,0,0), -1)
                    # drawing.draw_box(im, bbox, color=(0,255,0), thickness=2)

                    ex = ds3d.Pose3DExample(
                        image_path=im, camera=camera, bbox=bbox, mask=mask,
                        parameters=parameters, world_coords=None)

                    new_image_relpath = f'synbody_humannerf_downscaled/{impath}'
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        callback=examples.append)

    return ds3d.Pose3DDataset(ds3d.JointInfo([], []), examples)


def load_cameras(path):
    camdata = spu.load_json(path)
    return [
        cameralib.Camera(
            intrinsic_matrix=camdata[f'camera{i_cam:02d}']['K'],
            rot_world_to_cam=camdata[f'camera{i_cam:02d}']['R'],
            trans_after_rot=np.array(camdata[f'camera{i_cam:02d}']['T']) * 1000,
            world_up=(0, -1, 0))
        for i_cam in range(8)]


if __name__ == '__main__':
    make_dataset()
