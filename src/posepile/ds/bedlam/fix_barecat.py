import barecat
import simplepyutils as spu
from posepile.ds.agora.add_parametric import example_to_dict
from posepile.paths import CACHE_DIR, DATA_ROOT


def main():
    ds = spu.load_pickle(f'{CACHE_DIR}/bedlam2.pkl')

    with barecat.Barecat(
            f'{DATA_ROOT}/bc/merged_converted_smplx2.barecat', auto_codec=True,
            readonly=False, overwrite=True) as writer:
        with barecat.Barecat(
                f'{DATA_ROOT}/bc/merged_converted_smplx.barecat', decoder=decoder) as reader:

            for path in spu.progressbar(reader):
                if 'bedlam_downscaled' in path or 'behave_downscaled' in path:
                    continue

                data = reader[path]
                if 'parameters' in data:
                    data['parameters']['gender'] = 'neutral'
                    data['parameters']['type'] = 'smplx'
                writer[path] = data

        for phase_name, phase in zip('train val test'.split(), [0, 1, 2]):
            examples = ds.examples[phase]
            for i, ex in enumerate(spu.progressbar(examples)):
                if isinstance(ex.parameters, tuple):
                    ex.parameters = ex.parameters[0]
                writer[f'{phase_name}/{ex.image_path}_{0:02d}.msgpack'] = example_to_dict(ex)


if __name__ == '__main__':
    main()
