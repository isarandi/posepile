import glob
import os.path as osp

import bvhtoolbox  # https://github.com/OlafHaag/bvh-toolbox
import numpy as np
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    bvh_paths = glob.glob(f'{DATA_ROOT}/bmhad/Mocap/SkeletalData/*.bvh')
    spu.parallel_map_with_progbar(convert_to_npy, bvh_paths)


def convert_to_npy(bvh_path):
    with open(bvh_path) as f:
        bvh_tree = bvhtoolbox.BvhTree(f.read())

    coords_per_joint = []
    root = next(bvh_tree.root.filter('ROOT'))

    def get_world_positions(joint):
        if joint.value[0] == 'End':
            joint.world_transforms = np.tile(np.eye(4), (bvh_tree.nframes, 1, 1))
        else:
            channels = bvh_tree.joint_channels(joint.name)
            axes_order = ''.join([ch[:1] for ch in channels if ch[1:] == 'rotation']).lower()
            axes_order = 's' + axes_order[::-1]
            joint.world_transforms = bvhtoolbox.get_affines(bvh_tree, joint.name, axes=axes_order)

        if joint != root:
            offset = [float(o) for o in joint['OFFSET']]
            joint.world_transforms[:, :3, 3] = offset
            joint.world_transforms = joint.parent.world_transforms @ joint.world_transforms

        pos = joint.world_transforms[:, :3, 3]
        coords_per_joint.append(pos)

        end = list(joint.filter('End'))
        if end:
            get_world_positions(end[0])
        for child in joint.filter('JOINT'):
            get_world_positions(child)

    get_world_positions(root)
    coords = np.stack(coords_per_joint, axis=1) * 10
    out_path = osp.splitext(bvh_path)[0] + '.npy'
    np.save(out_path, coords)


if __name__ == '__main__':
    main()
