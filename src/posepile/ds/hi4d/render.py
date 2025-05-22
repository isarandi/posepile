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
import cameravision
import cv2
import imageio.v2 as imageio
import mathutils
import numpy as np
import rlemasklib
import simplepyutils as spu
from mathutils import Euler, Matrix, Vector
import os
import shutil

DATA_ROOT = '/work_uncached/sarandi/data'
RERENDER_DIR = f'{DATA_ROOT}/hi4d_rerender'
RESOLUTION = 512


def main():
    initialize_blender()
    seq_dirs = glob.glob(f'{DATA_ROOT}/hi4d/*/*')

    for pbar, seq_dir in spu.zip_progressbar(seq_dirs):
        seq = spu.path_range(seq_dir, -2, None)
        pbar.set_description(seq)
        if osp.exists(f'{RERENDER_DIR}/{seq}/cameras.pkl'):
            continue
        elif osp.exists(f'{RERENDER_DIR}/{seq}'):
            # Incomplete render, delete and rerender
            shutil.rmtree(f'{RERENDER_DIR}/{seq}')
        render_for_sequence(seq)


def render_for_sequence(seq):
    cameras = []
    masks = []
    metadata = np.load(f'{DATA_ROOT}/hi4d/{seq}/meta.npz')
    for i_image in spu.progressbar(range(1000)):
        i_frame = np.random.randint(int(metadata['start']), int(metadata['end']) + 1)
        mesh_path = f'{DATA_ROOT}/hi4d/{seq}/frames/mesh-f{i_frame:05d}.obj'

        smpl_data = np.load(f'{DATA_ROOT}/hi4d/{seq}/smpl/{i_frame:06d}.npz')
        i_person = np.random.randint(0, len(smpl_data['verts']))
        smpl_verts = smpl_data['verts'][i_person]
        person_center = (np.max(smpl_verts, axis=0) + np.min(smpl_verts, axis=0)) / 2
        person_size = np.max(np.max(smpl_verts, axis=0) - np.min(smpl_verts, axis=0))

        fov = np.random.uniform(30, 70)
        intrinsics = cameravision.intrinsics_from_fov(fov, [RESOLUTION, RESOLUTION])
        distance_to_person = 1.25 * person_size / (2.0 * math.tan(math.radians(fov) / 2.0))
        optical_center = person_center + random_dir() * distance_to_person
        camera = cameravision.Camera(
            optical_center=optical_center, intrinsic_matrix=intrinsics, world_up=(0, 1, 0))
        camera.turn_towards(target_world_point=person_center)

        with silence_output():
            rgb, mask = render_image(mesh_path, camera, person_center)
        jpeg_path = f'{RERENDER_DIR}/{seq}/f_{i_image:06d}_{i_frame:06d}_{i_person}.jpg'
        spu.ensure_parent_dir_exists(jpeg_path)
        imageio.imwrite(jpeg_path, rgb, quality=95)

        camera.t *= 1000
        cameras.append(camera)
        masks.append(rlemasklib.encode(mask))
    spu.dump_pickle(cameras, f'{RERENDER_DIR}/{seq}/cameras.pkl')
    spu.dump_pickle(masks, f'{RERENDER_DIR}/{seq}/masks.pkl')


def render_image(mesh_path, camera, person_center):
    clear_objects()

    bpy.ops.wm.obj_import(filepath=mesh_path)

    target = bpy.context.selected_objects[0]
    # Add monkeys optionally
    # for i in range(10):
    #     bpy.ops.mesh.primitive_monkey_add(
    #         size=np.random.uniform(0.1, 1), align='WORLD',
    #         location=person_center + np.random.uniform(0, 2) * Vector(tuple(random_dir(
    #         ))))
    #     bpy.context.active_object.data.materials.append(transparent_material)

    # Add camera
    blender_camera = bpy.data.objects.new('RandomCamera', bpy.data.cameras.new('RandomCamera'))
    bpy.context.collection.objects.link(blender_camera)
    bpy.context.scene.camera = blender_camera
    set_blender_cam(blender_camera, camera)

    # Add lights
    num_lights = np.random.randint(5, 15)
    person_center = Vector(list(person_center))

    for i in range(num_lights):
        light = bpy.data.objects.new(
            name='RandomLight', object_data=bpy.data.lights.new(name='RandomLight', type='POINT'))
        light.data.color = np.random.uniform(size=3)
        bpy.context.collection.objects.link(light)
        light.location = target.matrix_world @ (
                person_center + np.random.uniform(3, 10) * Vector(tuple(random_dir())))
        light.data.energy = random.uniform(200, 600)

    area_light = bpy.data.objects.new(
        name="AreaLight",
        object_data=bpy.data.lights.new(name="AreaLight", type='AREA'))
    area_light.data.shape = 'DISK'
    area_light.data.size = random.uniform(0.5, 5)
    area_light.data.energy = random.uniform(0, 1500)
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


def set_blender_cam(blender_camera, cameravision_camera):
    fx = cameravision_camera.intrinsic_matrix[0, 0]
    assert np.all(np.isclose(
        np.array([[fx, 0], [0, fx]]),
        cameravision_camera.intrinsic_matrix[:2, :2]))

    extrinsics = cameravision_camera.get_extrinsic_matrix()
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


#
# def create_glassy_material():
#     # Create a new material
#     mat = bpy.data.materials.new(name="GlassMaterial")
#
#     # Enable the use of shader nodes
#     mat.use_nodes = True
#
#     # Get the shader node tree of the material
#     node_tree = mat.node_tree
#     nodes = node_tree.nodes
#
#     # Clear existing nodes
#     for node in nodes:
#         nodes.remove(node)
#
#     # Add a ShaderNodeBsdfGlass node
#     glass_node = nodes.new(type='ShaderNodeBsdfGlass')
#
#     # Add a ShaderNodeOutputMaterial node
#     output_node = nodes.new(type='ShaderNodeOutputMaterial')
#
#     # Connect the nodes
#     node_tree.links.new(glass_node.outputs['BSDF'], output_node.inputs['Surface'])
#
#     # Set properties of the ShaderNodeBsdfGlass
#     glass_node.distribution = 'GGX'  # You can choose 'BECKMANN', 'GGX', or 'MULTI_GGX'
#     glass_node.inputs['IOR'].default_value = 1.01
#     return mat
#
#
# # Function to create a transparent material
# def create_transparent_material():
#     mat = bpy.data.materials.new(name="InvisibleMaterial")
#     mat.use_nodes = True
#     nodes = mat.node_tree.nodes
#     nodes.clear()  # Clear existing nodes
#
#     # Create necessary nodes
#     transparent_bsdf = nodes.new(type='ShaderNodeBsdfTransparent')
#     material_output = nodes.new(type='ShaderNodeOutputMaterial')
#
#     # Connect nodes
#     links = mat.node_tree.links
#     links.new(transparent_bsdf.outputs[0], material_output.inputs[0])
#
#     # Set material to be transparent in Eevee
#     mat.blend_method = 'CLIP'
#
#     return mat
#
#
# def create_invisible_shadow_material():
#     # Create a new material
#     material = bpy.data.materials.new(name="Semi-Transparent")
#
#     # Use nodes for the material
#     material.use_nodes = True
#     nodes = material.node_tree.nodes
#     nodes.clear()
#
#     # Create a Transparent shader
#     transparent_shader = nodes.new(type='ShaderNodeBsdfTransparent')
#
#     # Create a Shader to RGB node
#     shader_to_rgb = nodes.new(type='ShaderNodeShaderToRGB')
#
#     # Create a Mix Shader node
#     mix_shader = nodes.new(type='ShaderNodeMixShader')
#
#     # Create a Material Output node
#     material_output = nodes.new(type='ShaderNodeOutputMaterial')
#
#     # Link the nodes
#     links = material.node_tree.links
#     links.new(transparent_shader.outputs['BSDF'], mix_shader.inputs[1])
#     links.new(shader_to_rgb.outputs['Color'], mix_shader.inputs[0])
#     links.new(mix_shader.outputs['Shader'], material_output.inputs['Surface'])
#
#     # Set blend mode to CLIP for full transparency
#     material.blend_method = 'CLIP'
#
#     # Adjust the Mix Shader factor to control the level of transparency
#     mix_shader.inputs['Fac'].default_value = 0.05  # Adjust this value as needed
#
#     return material


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
