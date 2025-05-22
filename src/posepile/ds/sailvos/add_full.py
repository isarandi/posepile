import functools
import glob
import os.path as osp
import sys

import barecat
import numpy as np
import posepile.ds.sailvos.main as sailvos_main
import posepile.joint_info
import simplepyutils as spu
from posepile.ds.sailvos.main import INTERESTING_JOINTS
from posepile.paths import CACHE_DIR, DATA_ROOT
from posepile.tools.dataset_pickle_to_barecat import example_to_dict

sys.modules['data.joint_info'] = posepile.joint_info
sys.modules['data'] = posepile


def main():
    sailvos_root = f'{DATA_ROOT}/sailvos'
    ds = spu.load_pickle(f'{CACHE_DIR}/sailvos.pkl')

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/sailvos.barecat', overwrite=True, readonly=False,
            auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        cached_loadtxt = functools.lru_cache(maxsize=1000)(np.loadtxt)
        impath_to_objname = {}
        for phase_name, phase in zip('train val test'.split(), [0, 1, 2]):
            examples = ds.examples[phase]
            for i, ex in enumerate(spu.progressbar(examples)):
                seq_name, i_frame, i_person = parse_cropped_imagepath(ex.image_path)
                filepaths3d = glob.glob(f'{sailvos_root}/{seq_name}/*/{i_frame}_3d.txt')

                for filepath3d in filepaths3d:
                    all_coords3d = cached_loadtxt(filepath3d, dtype=np.float32) * 1000
                    all_coords3d[all_coords3d == 0] = np.nan
                    world_coords = np.array([
                        np.nanmean(all_coords3d[alts], axis=0) for alts in INTERESTING_JOINTS])
                    camera = sailvos_main.load_camera(
                        filepath3d.replace('_3d.txt', '_camera.npz'))
                    if camera is None:
                        continue

                    world_coords += ex.camera.t - camera.t
                    if np.all(
                            np.isclose(world_coords, ex.world_coords,
                                       rtol=0, atol=1, equal_nan=True)):
                        break
                else:
                    raise ValueError('No matching 3D file found')

                objname = spu.split_path(filepath3d)[-2]
                impath_to_objname[ex.image_path] = objname
                ex.world_coords = all_coords3d + (ex.camera.t - camera.t)
                bc_writer[f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)
    spu.dump_pickle(impath_to_objname, f'{DATA_ROOT}/sailvos/impath_to_objname.pkl')


@functools.lru_cache(maxsize=1000)
def cached_load_camera(path):
    a = np.load(path)
    if np.all(np.isnan(a['intrinsics'])):
        return None

    cam = cameralib.Camera(extrinsic_matrix=a['extrinsics'], intrinsic_matrix=a['intrinsics'])
    cam.t *= 1000
    return cam


def parse_cropped_imagepath(p):
    filename = osp.basename(p)
    seq_name = spu.split_path(p)[1]
    noext = osp.splitext(filename)[0]
    parts = noext.split('_')
    firstpart = '_'.join(parts[:-1])
    person_id = int(parts[-1])
    return seq_name, firstpart, person_id


if __name__ == '__main__':
    main()
