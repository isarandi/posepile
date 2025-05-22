import argparse

import barecat
import numpy as np
import simplepyutils as spu
from simplepyutils import FLAGS

from posepile.joint_info import JointInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--joint-info-path', type=str)
    spu.initialize(parser)
    ji_new = spu.load_pickle(FLAGS.joint_info_path)

    with (barecat.Barecat(FLAGS.input_path, auto_codec=True) as bc_reader,
          barecat.Barecat(
              FLAGS.output_path, overwrite=True, auto_codec=True,
              readonly=False) as bc_writer):

        metadata_old = bc_reader['metadata.msgpack']
        ji_old = JointInfo(
            [n + '_coco' for n in metadata_old['joint_names']], metadata_old['joint_edges'])

        bc_writer['metadata.msgpack'] = dict(
            joint_names=ji_new.names,
            joint_edges=ji_new.stick_figure_edges)

        indices_new = np.array([ji_new.names.index(n) for n in ji_old.names if n in ji_new.names])
        i_selector = np.array([i for i, n in enumerate(ji_old.names) if n in ji_new.names])
        # dict(rows=valid_coords, i_rows=np.asarray(i_valid_joints, np.uint16))
        for name, data in spu.progressbar_items(bc_reader):
            if name == 'metadata.msgpack':
                continue

            old_valids = data['joints3d']['rows']
            i_old_valids = data['joints3d']['i_rows']
            if len(i_old_valids) == 0:
                i_old_valids = range(len(old_valids))

            i_old_of_kept_valids = [i_old for i_old in i_selector if i_old in i_old_valids]
            i_among_old_valids_of_kept_valids = [
                i_among_valids for i_among_valids, i_old in enumerate(i_old_valids)
                if i_old in i_old_of_kept_valids]

            new_valids = np.ascontiguousarray(old_valids[i_among_old_valids_of_kept_valids])
            i_new_valids = indices_new[i_old_of_kept_valids]
            if i_new_valids.shape[0] == ji_new.n_joints:
                i_new_valids = i_new_valids[:0]

            data['joints3d']['rows'] = new_valids
            data['joints3d']['i_rows'] = np.uint16(i_new_valids)
            bc_writer[name] = data


if __name__ == '__main__':
    main()
