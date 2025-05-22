import argparse
import os.path as osp
import sys

import simplepyutils as spu
from simplepyutils import FLAGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    spu.initialize(parser)
    update_pickle_file(FLAGS.path, FLAGS.path, 'data')


def update_pickle_file(pickle_in_path, pickle_out_path, legacy_module_name):
    if legacy_module_name:
        import posepile
        import posepile.joint_info
        import posepile.datasets3d
        import posepile.datasets2d
        sys.modules[f"{legacy_module_name}.datasets3d"] = posepile.datasets3d
        sys.modules[f"{legacy_module_name}.datasets2d"] = posepile.datasets2d
        sys.modules[f"{legacy_module_name}.joint_info"] = posepile.joint_info
        sys.modules[f"{legacy_module_name}"] = posepile
    ds = spu.load_pickle(pickle_in_path)
    fix_legacy_issues(ds)
    spu.dump_pickle(ds, pickle_out_path)


def fix_legacy_issues(ds):
    def fix_path(p):
        if osp.isabs(p):
            return osp.normpath(osp.relpath(p, '/nodes/brewdog/work3/sarandi/data_reprod'))
        else:
            return osp.normpath(p)

    for ex in spu.progressbar(ds.iter_examples()):
        ex.image_path = fix_path(ex.image_path)
        if hasattr(ex, 'mask') and isinstance(ex.mask, list):
            assert len(ex.mask) == 1 or all(m == ex.mask[0] for m in ex.mask)
            ex.mask = ex.mask[0]

if __name__ == '__main__':
    main()