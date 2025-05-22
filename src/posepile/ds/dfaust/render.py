import contextlib
import math
import os
import os.path as osp
import random
import sys

import bpy
import cameralib
import h5py
import imageio.v2 as imageio
import numpy as np
import rlemasklib
import scipy.sparse as sps
import simplepyutils as spu
import trimesh
from mathutils import Matrix, Vector

DATA_ROOT = '/work/sarandi/data'
DATASET_DIR = f'{DATA_ROOT}/dfaust'
RERENDER_DIR = f'{DATA_ROOT}/dfaust_render'
RESOLUTION = 512


def main():
    initialize_blender()

    registrations = dict(
        m=h5py.File(f'{DATASET_DIR}/registrations_m.hdf5', 'r'),
        f=h5py.File(f'{DATASET_DIR}/registrations_f.hdf5', 'r'))
    texture_paths_all = spu.sorted_recursive_glob(f'{DATA_ROOT}/smplitex/*.png')
    genders = spu.read_file(f'{DATA_ROOT}/smplitex/genders.txt')
    mesh_saver = SMPLMeshSaver()

    texture_paths = dict(m=[], f=[])
    for texture_path, gender in zip(texture_paths_all, genders):
        texture_paths[gender].append(texture_path)

    for g in 'mf':
        for key in registrations[g].keys():
            if key == 'faces':
                continue

            verts_seq = registrations[g][key][:].transpose(2, 0, 1)
            n_frames = len(verts_seq)

            cameras = []
            masks = []

            for i_gen in spu.progressbar(range(200)):
                i_frame = np.random.randint(n_frames)
                verts = verts_seq[i_frame]

                texture_path = random.choice(texture_paths[g])
                mesh_saver.save(verts_seq[i_frame])
                if osp.exists(f'{RERENDER_DIR}/texture.png'):
                    os.unlink(f'{RERENDER_DIR}/texture.png')
                os.symlink(texture_path, f'{RERENDER_DIR}/texture.png')

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
                    rgb, mask = render_image(mesh_saver.out_path, camera, person_center)
                jpeg_path = f'{RERENDER_DIR}/{key}/{i_gen:06d}_{i_frame:06d}.jpg'
                spu.ensure_parent_dir_exists(jpeg_path)
                imageio.imwrite(jpeg_path, rgb, quality=95)
                camera.t *= 1000
                cameras.append(camera)
                masks.append(rlemasklib.encode(mask))

            spu.dump_pickle(cameras, f'{RERENDER_DIR}/{key}/cameras.pkl')
            spu.dump_pickle(masks, f'{RERENDER_DIR}/{key}/masks.pkl')


class SMPLMeshSaver:
    def __init__(self):
        self.m_split = trimesh.load(f'{RERENDER_DIR}/smpl_uv.obj')
        split_verts = np.array(self.m_split.vertices, copy=True)
        merged_verts = np.load(f'{RERENDER_DIR}/smpl_default_verts_f.npy')
        merged_to_split_matrix = sps.lil_matrix((len(split_verts), len(merged_verts)))
        for i, v in enumerate(split_verts):
            merged_to_split_matrix[
                i, np.argmin(np.linalg.norm(merged_verts - v, axis=-1))] = 1
        self.merged_to_split_matrix = merged_to_split_matrix.tocsr()
        self.out_path = f'{RERENDER_DIR}/smpl_posed_uv.obj'

    def save(self, verts):
        self.m_split.vertices = self.merged_to_split_matrix @ verts
        spu.write_file(self.m_split.export('obj'), self.out_path)


def render_image(mesh_path, camera, person_center):
    clear_objects()
    bpy.ops.wm.obj_import(filepath=mesh_path)

    target = bpy.context.selected_objects[0]
    bpy.ops.object.shade_smooth()
    target.data.use_auto_smooth = True
    target.data.auto_smooth_angle = np.deg2rad(60)

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
