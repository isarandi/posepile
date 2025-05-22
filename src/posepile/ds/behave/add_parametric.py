import os.path as osp
import sys

import barecat
import numpy as np
import posepile.joint_info
import simplepyutils as spu
from posepile.ds.agora.add_parametric import example_to_dict
from posepile.paths import CACHE_DIR, DATA_ROOT

sys.modules['data.joint_info'] = posepile.joint_info
sys.modules['data'] = posepile


def main():
    ds = spu.load_pickle(f'{CACHE_DIR}/behave.pkl')
    behave_root = f'{DATA_ROOT}/behave'

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/behave_smplh.barecat',
            overwrite=True, readonly=False, auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        for phase_name, phase in zip('train val test'.split(), [0, 1, 2]):
            examples = ds.examples[phase]
            for i, ex in enumerate(spu.progressbar(examples)):
                seq_dir = osp.join(behave_root, spu.path_range(ex.image_path, 1, 3))
                gender = spu.load_json(f'{seq_dir}/info.json')['gender']

                fit_path = osp.join(behave_root, spu.path_range(ex.image_path, 1, 4),
                                    'person/fit02/person_fit.pkl')
                data = spu.load_pickle(fit_path)
                ex.parameters = dict(
                    type='smplh', gender=gender, pose=data['pose'],
                    shape=data['betas'], kid_factor=np.float32(0),
                    trans=data['trans'])
                bc_writer[
                    f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


if __name__ == '__main__':
    main()
