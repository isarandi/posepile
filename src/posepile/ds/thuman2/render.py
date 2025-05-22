import contextlib
import functools
import glob
import io
import math
import os
import os.path as osp
import random
import sys
from contextlib import contextmanager

import bpy
import cameralib
import cv2
import imageio.v2 as imageio
import mathutils
import numpy as np
import rlemasklib
import simplepyutils as spu
from mathutils import Euler, Matrix, Vector
import os
import shutil
import smpl.numpy

DATA_ROOT = '/work/sarandi/data'
RERENDER_DIR = f'{DATA_ROOT}/thuman2_render'
RESOLUTION = 512


def main():
    initialize_blender()
    smplx_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/thuman2/smplx/*/smplx_param.pkl')

    for pbar, smplx_path in spu.zip_progressbar(smplx_paths):
        i_scan = osp.basename(osp.dirname(smplx_path))
        pbar.set_description(i_scan)

        if osp.exists(f'{RERENDER_DIR}/{i_scan}/cameras.pkl'):
            continue
        elif osp.exists(f'{RERENDER_DIR}/{i_scan}'):
            # Incomplete render, delete and rerender
            shutil.rmtree(f'{RERENDER_DIR}/{i_scan}')
        render_for_scan(i_scan, smplx_path)


def render_for_scan(i_scan, smplx_path):
    cameras = []
    masks = []
    smplx_bm = smpl.numpy.get_cached_body_model('smplx', 'male')
    mesh_path = f'{DATA_ROOT}/thuman2/{i_scan}/{i_scan}.obj'

    for i_image in spu.progressbar(range(100)):
        pose, shape, trans, expression, scale = load_smplx_params(smplx_path, gender='male')
        verts = smplx_bm.single(pose, shape, trans)['vertices'] * scale
        person_center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2
        person_size = np.max(np.max(verts, axis=0) - np.min(verts, axis=0))

        fov = np.random.uniform(30, 70)
        intrinsics = cameralib.intrinsics_from_fov(fov, [RESOLUTION, RESOLUTION])
        distance_to_person = 1.25 * person_size / (2.0 * math.tan(math.radians(fov) / 2.0))
        optical_center = person_center + random_dir() * distance_to_person
        camera = cameralib.Camera(
            optical_center=optical_center, intrinsic_matrix=intrinsics, world_up=(0, 1, 0))
        camera.turn_towards(target_world_point=person_center)

        with silence_output():
            rgb, mask = render_image(mesh_path, camera, person_center)
        jpeg_path = f'{RERENDER_DIR}/{i_scan}/{i_image:02d}.jpg'
        spu.ensure_parent_dir_exists(jpeg_path)

        # verts2d = camera.world_to_image(verts)
        # for vert2d in verts2d:
        #     cv2.circle(rgb, tuple(vert2d.astype(np.int32)), 2, (0, 255, 0), -1)

        imageio.imwrite(jpeg_path, rgb, quality=95)

        camera.t *= 1000
        cameras.append(camera)
        masks.append(rlemasklib.encode(mask))
    spu.dump_pickle(cameras, f'{RERENDER_DIR}/{i_scan}/cameras.pkl')
    spu.dump_pickle(masks, f'{RERENDER_DIR}/{i_scan}/masks.pkl')


def render_image(mesh_path, camera, person_center):
    clear_objects()
    bpy.ops.wm.obj_import(filepath=mesh_path)

    target = bpy.context.selected_objects[0]
    # Add camera
    blender_camera = bpy.data.objects.new('RandomCamera', bpy.data.cameras.new('RandomCamera'))
    bpy.context.collection.objects.link(blender_camera)
    bpy.context.scene.camera = blender_camera
    set_blender_cam(blender_camera, camera)

    # Add lights
    num_lights = np.random.randint(5, 10)
    person_center = Vector(list(person_center))

    for i in range(num_lights):
        light = bpy.data.objects.new(
            name='RandomLight', object_data=bpy.data.lights.new(name='RandomLight', type='POINT'))
        light.data.color = np.random.uniform(size=3)
        bpy.context.collection.objects.link(light)
        light.location = target.matrix_world @ (
                person_center + np.random.uniform(3, 10) * Vector(tuple(random_dir())))
        light.data.energy = random.uniform(200, 400)

    area_light = bpy.data.objects.new(
        name="AreaLight",
        object_data=bpy.data.lights.new(name="AreaLight", type='AREA'))
    area_light.data.shape = 'DISK'
    area_light.data.size = random.uniform(0.5, 5)
    area_light.data.energy = random.uniform(0, 1000)
    area_light.data.color = (1, 1, 1)

    if random.uniform(0, 1) < 0.8:
        area_light.location = target.matrix_world @ (person_center + Vector((0, 3, 0)))
        area_light.rotation_euler = (0, 0, 0)
    else:
        area_light.location = target.matrix_world @ (person_center + Vector((0, -3, 0)))
        area_light.rotation_euler = (math.pi, 0, 0)

    bpy.context.collection.objects.link(area_light)

    # Actual render
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)

    # Read back the rendered image and mask
    rgba = imageio.imread('/tmp/output.png')
    rgb = rgba[:, :, :3]
    mask = np.uint8(rgba[:, :, 3] > 127) * 255
    alpha = rgba[:, :, 3:].astype(np.float32) / 255
    rgb = (
            rgb.astype(np.float32) * alpha +
            np.array([127, 127, 127], np.float32) * (1 - alpha))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, mask


def set_blender_cam(blender_camera, cameralib_camera):
    fx = cameralib_camera.intrinsic_matrix[0, 0]
    assert np.all(np.isclose(
        np.array([[fx, 0], [0, fx]]),
        cameralib_camera.intrinsic_matrix[:2, :2]))

    extrinsics = cameralib_camera.get_extrinsic_matrix()
    R = extrinsics[:3, :3]
    T = extrinsics[:3, 3]
    R1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)
    BR = R2 @ R.T @ R1
    BT = -BR @ R1.T @ T
    B = np.concatenate([
        np.concatenate([BR, BT[:, np.newaxis]], axis=1),
        np.array([[0, 0, 0, 1]], np.float32)], axis=0)
    blender_camera.matrix_world = Matrix(B.tolist())
    blender_camera.data.lens = fx / RESOLUTION * blender_camera.data.sensor_width


def random_dir():
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(math.pi / 6, math.pi / 2 + math.pi / 6)
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return np.array([x, z, y], np.float32)


@contextlib.contextmanager
def _silence(which):
    the_fd = which.fileno()
    orig_fd = os.dup(the_fd)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, the_fd)
    try:
        yield
    finally:
        os.dup2(orig_fd, the_fd)
        os.close(orig_fd)
        os.close(null_fd)


@contextlib.contextmanager
def silence_output():
    with _silence(sys.stdout), _silence(sys.stderr):
        yield


def clear_objects():
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.outliner.orphans_purge()


def load_smplx_params(path, gender=None):
    data = spu.load_pickle(path)
    get = lambda x: data[x][0].astype(np.float32)
    pose_parts = [
        'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',
    ]
    pose = np.concatenate([get(x).reshape(-1) for x in pose_parts], axis=0)
    left_hand_mean, right_hand_mean, left_hand_mat, right_hand_mat = get_hand_pca(gender)
    left_hand_components = get('left_hand_pose')
    right_hand_components = get('right_hand_pose')
    left_hand_pose = (
            left_hand_components @ left_hand_mat[:left_hand_components.shape[-1]] +
            left_hand_mean)
    right_hand_pose = (
            right_hand_components @ right_hand_mat[:right_hand_components.shape[-1]] +
            right_hand_mean)
    pose = np.concatenate([pose, left_hand_pose, right_hand_pose], axis=0)
    scale = get('scale')
    return (
        pose, get('betas'), np.array(data['translation'], np.float32) / scale, get('expression'),
        scale)


@functools.lru_cache()
def get_hand_pca(gender):
    gender_map = dict(f='FEMALE', m='MALE', n='NEUTRAL')
    a = np.load(f'{DATA_ROOT}/body_models/smplx/SMPLX_{gender_map[gender[0].lower()]}.npz')
    return a['hands_meanl'], a['hands_meanr'], a['hands_componentsl'], a['hands_componentsr']


def initialize_blender():
    bpy.context.scene.render.resolution_x = RESOLUTION
    bpy.context.scene.render.resolution_y = RESOLUTION
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = '/tmp/output.png'
    bpy.context.scene.cycles.samples = 512
    bpy.context.scene.cycles.use_soft_shadows = True
    bpy.context.scene.render.film_transparent = True
    # Set the device_type
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = '3090' in d["name"]


if __name__ == '__main__':
    main()
