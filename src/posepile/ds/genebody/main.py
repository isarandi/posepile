import os.path as osp

import cameralib
import numpy as np
import posepile.datasets3d as ds3d
import rlemasklib
import simplepyutils as spu
import smpl.numpy
from posepile.ds.rich.add_parametric import load_smplx_params
from posepile.paths import DATA_ROOT
from posepile.util.adaptive_pose_sampling import AdaptivePoseSampler2
from posepile.util.preproc_for_efficiency import make_efficient_example
from posepile.joint_info import JointInfo

DATASET_NAME = 'genebody'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


@spu.picklecache(f'{DATASET_NAME}.pkl', min_time="2023-12-01T13:35:22")
def make_dataset():
    examples = []
    split = np.load(
        f'{DATASET_DIR}/GeneBody-Train40/genebody_split.npy', allow_pickle=True).item()
    train_names = split['train']
    mask_dict = spu.load_pickle(f'{DATASET_DIR}/masks.pkl')

    with spu.ThrottledPool() as pool:
        for name in train_names:
            person_dir = f'{DATASET_DIR}/GeneBody-Train40/{name}'
            param_paths = spu.sorted_recursive_glob(f'{person_dir}/param/*.npy')
            frame_ids = [osp.splitext(osp.basename(p))[0] for p in param_paths]
            cameras = load_cameras(f'{person_dir}/annots.npy')
            sampler = AdaptivePoseSampler2(100, True, True, 100)

            for frame_id in spu.progressbar(frame_ids, desc=name, unit=' frames'):
                smplx_data = np.load(
                    f'{person_dir}/param/{frame_id}.npy', allow_pickle=True).item()
                gender = name_to_gender[name]
                pose, betas, trans, kid_factor, expression = load_smplx_params(
                    smplx_data['smplx'], gender)
                parameters = dict(
                    type='smplx', gender=gender, pose=pose, shape=betas, expression=expression,
                    scale=np.float32(smplx_data['smplx_scale']), kid_factor=np.float32(0),
                    trans=trans)

                joints = smpl.numpy.get_cached_body_model(
                    'smplx', gender).single(
                    pose, betas, trans, return_vertices=False)['joints'] * smplx_data['smplx_scale']
                if sampler.should_skip(joints * 1000):
                    continue

                for cam_id, camera in cameras.items():
                    mask = mask_dict[
                        f'GeneBody-Train40/{name}/mask/{cam_id}/mask{frame_id}.png']
                    bbox = rlemasklib.to_bbox(mask)
                    impath = f'{person_dir}/image/{cam_id}/{frame_id}.jpg'
                    ex = ds3d.Pose3DExample(
                        image_path=impath, camera=camera, bbox=bbox, mask=mask,
                        parameters=parameters, world_coords=None)

                    new_image_relpath = (
                        f'genebody_downscaled/{name}/image/{cam_id}/{frame_id}.jpg')
                    pool.apply_async(
                        make_efficient_example, (ex, new_image_relpath),
                        kwargs=dict(
                            downscale_input_for_antialias=True, downscale_at_decode=False,
                            reuse_image_array=True),
                        callback=examples.append)

    return ds3d.Pose3DDataset(JointInfo([], []), examples)


def load_cameras(path):
    camdata = np.load(path, allow_pickle=True).item()['cams']
    result = {k: cameralib.Camera(
        extrinsic_matrix=np.linalg.inv(d['c2w']), intrinsic_matrix=d['K'],
        distortion_coeffs=d['D'], world_up=(0, -1, 0))
        for k, d in camdata.items()}
    for c in result.values():
        c.t *= 1000
    return result


name_to_gender = {
    'abror': 'male', 'ahha': 'female', 'alejandro': 'male', 'amanda': 'female', 'amaris': 'female',
    'anastasia': 'female', 'aosilan': 'male', 'arslan': 'male', 'barlas': 'male', 'barry': 'male',
    'camilo': 'male', 'dannier': 'male', 'dilshod': 'male', 'fenghaohan': 'male',
    'fuzhizhi': 'female', 'fuzhizhi2': 'female', 'gaoxing': 'female', 'huajiangtao3': 'male',
    'huajiangtao5': 'male', 'ivan1': 'male', 'ivan2': 'male',
    'jinyutong': 'female', 'jinyutong2': 'female',
    'joseph_matanda': 'female', 'kamal_ejaz': 'male', 'kemal': 'male', 'lihongyun': 'female',
    'mahaoran': 'male', 'maria': 'female', 'natacha': 'female', 'quyuanning': 'female',
    'rivera': 'male', 'shchyerbina_oleksandr': 'male',
    'soufianou_boubacar_moumouni': 'male', 'sunyuxing': 'male', 'Tichinah_jervier': 'female',
    'wangxiang': 'male', 'wuwenyan': 'male', 'xujiarui': 'female', 'yaoqibin': 'male',
    'zhanghao': 'male', 'zhanghongwei': 'female', 'zhangzixiao': 'female', 'zhengxin': 'female',
    'zhonglantai': 'female', 'zhuna': 'female', 'zhuna2': 'female', 'zhuxuezhi': 'male',
    'songyujie': 'male', 'rabbi': 'male', 'zhangziyu': 'female'}

if __name__ == '__main__':
    make_dataset()
