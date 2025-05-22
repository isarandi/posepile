import barecat
import simplepyutils as spu

from posepile.ds.agora.add_parametric import example_to_dict
from posepile.paths import CACHE_DIR, DATA_ROOT


def main():
    ds = spu.load_pickle(f'{CACHE_DIR}/moyo.pkl')
    with barecat.Barecat(
            f'{DATA_ROOT}/bc/moyo.barecat',
            overwrite=True, readonly=False, auto_codec=True) as bc_writer:
        bc_writer['metadata.msgpack'] = dict(
            joint_names=[], joint_edges=[], train_bone_lengths=[], trainval_bone_lengths=[])

        for phase_name, phase in zip('train val test'.split(), [0, 1, 2]):
            examples = ds.examples[phase]
            for i, ex in enumerate(spu.progressbar(examples)):
                bc_writer[f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


if __name__ == '__main__':
    main()
