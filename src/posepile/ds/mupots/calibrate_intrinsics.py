import numpy as np
import simplepyutils as spu
from posepile import util
from posepile.paths import DATA_ROOT


def main():
    intrinsics_per_sequence = {}
    for i_seq in range(1, 21):
        anno = util.load_mat(f'{DATA_ROOT}/mupots/TS{i_seq}/annot.mat').annotations
        points2d = np.concatenate([x.annot2.T for x in np.nditer(anno) if x.isValidFrame])
        points3d = np.concatenate([x.annot3.T for x in np.nditer(anno) if x.isValidFrame])
        intrinsics_per_sequence[f'TS{i_seq}'] = estimate_intrinsic_matrix(points2d, points3d)

    spu.dump_json(intrinsics_per_sequence, f'{DATA_ROOT}/mupots/camera_intrinsics.json')


def estimate_intrinsic_matrix(points2d, points3d):
    n_rows = len(points2d) * 2
    A = np.empty((n_rows, 4))
    b = np.empty((n_rows, 1))
    for i, ((x2, y2), (x3, y3, z3)) in enumerate(zip(points2d, points3d)):
        A[2 * i] = [x3 / z3, 0, 1, 0]
        A[2 * i + 1] = [0, y3 / z3, 0, 1]
        b[2 * i] = [x2]
        b[2 * i + 1] = [y2]
    fx, fy, cx, cy = np.linalg.lstsq(A, b, rcond=None)[0][:, 0]
    return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]


if __name__ == '__main__':
    main()
