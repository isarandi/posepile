import glob
import os
import os.path as osp
import xml.etree.ElementTree

import cameralib
import numpy as np
import simplepyutils as spu

from posepile.paths import DATA_ROOT


def main():
    CWI_ROOT = f'{DATA_ROOT}/cwi'
    all_cameras = {}
    for path in glob.glob(f'{CWI_ROOT}/*/*/cameraconfig.xml'):
        dirname_rel = osp.relpath(osp.dirname(path), CWI_ROOT)
        cameras = load_cameras(path)
        for relpath, camera in cameras.items():
            video_relpath = f'{dirname_rel}/{relpath}'
            all_cameras[video_relpath] = camera

    spu.dump_pickle(all_cameras, f'{CWI_ROOT}/cameras.pkl')


def load_cameras(path):
    root = xml.etree.ElementTree.parse(path).getroot()
    camera_elems = root.find('CameraConfig').findall('camera')

    cameras = {}
    dirname = osp.dirname(path)
    for camera_elem in camera_elems:
        filename = camera_elem.attrib['filename']
        intr_mat, dist_coeffs = load_intrinsics(f'{dirname}/{filename}_calibration.json')
        t = camera_elem.find('trafo').find('values').attrib
        extr = np.linalg.inv(np.array([
            [t['v00'], t['v01'], t['v02'], t['v03']],
            [t['v10'], t['v11'], t['v12'], t['v13']],
            [t['v20'], t['v21'], t['v22'], t['v23']],
            [t['v30'], t['v31'], t['v32'], t['v33']]]).astype(np.float32))
        extr[:3, 3] *= 1000
        cameras[filename] = cameralib.Camera(
            extrinsic_matrix=extr, intrinsic_matrix=intr_mat, world_up=(0, 1, 0),
            distortion_coeffs=dist_coeffs)
    return cameras


def load_intrinsics(path):
    cameras = spu.load_json(path)['CalibrationInformation']['Cameras']
    camera_json = next(
        c for c in cameras if c['Purpose'] == 'CALIBRATION_CameraPurposePhotoVideo')
    cx, cy, fx, fy, k1, k2, k3, k4, k5, k6, codx, cody, p2, p1 = camera_json['Intrinsics'][
        'ModelParameters']
    intr = np.array([[fx * 2048, 0, cx * 2048], [0, fy * 1536, cy * 1536], [0, 0, 1]], np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], np.float32)
    return intr, dist_coeffs


if __name__ == '__main__':
    main()
