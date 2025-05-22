import os.path as osp

import simplepyutils as spu

from posepile.ds.egoexo4d.main import load_cameras
from posepile.paths import DATA_ROOT

DATASET_NAME = 'egoexo4d'
DATASET_DIR = f'{DATA_ROOT}/{DATASET_NAME}'


def main():
    take_data = spu.load_json(f'{DATASET_DIR}/takes.json')
    take_uid_to_phase = spu.load_json(f'{DATASET_DIR}/annotations/splits.json')['take_uid_to_split']

    take_campaths = [
        (f'{DATASET_DIR}/annotations/ego_pose/{take_uid_to_phase[take["take_uid"]]}/'
         f'camera_pose/{take["take_uid"]}.json') for take in take_data]

    cam_dicts = spu.parallel_map_with_progbar(get_cams_of_take, take_campaths)
    cameras_all = {}

    for takeinfo, cam_dict in zip(take_data, cam_dicts):
        vid_dir = f'takes/{takeinfo["take_name"]}/frame_aligned_videos'
        for camname, cam in cam_dict.items():
            cameras_all[f'{vid_dir}/{camname}.mp4'] = cam
            cameras_all[f'{vid_dir}/downscaled/448/{camname}.mp4'] = cam.scale_output(
                448 / 2160, inplace=False)

    spu.dump_pickle(cameras_all, f'{DATASET_DIR}/cameras.pkl')


def get_cams_of_take(take_campath):
    if not osp.exists(take_campath):
        return {}

    return load_cameras(take_campath)


if __name__ == '__main__':
    main()
