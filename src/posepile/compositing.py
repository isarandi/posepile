import os.path as osp

import boxlib
import cameravision
import imageio.v2 as imageio
import numpy as np
import rlemasklib
import scipy.optimize
import simplepyutils as spu

import posepile.datasets3d as ds3d
import posepile.util.improc as improc
from posepile import util
from posepile.paths import DATA_ROOT


def make_composited_dataset(stage1_ds, detections):
    grouped_by_img = spu.groupby(stage1_ds.examples[0], lambda ex: ex.image_path)
    yolo_examples = []
    for image_path, gt_people in spu.progressbar(grouped_by_img.items()):
        image_filename = osp.basename(image_path)
        boxes = [box for box in detections[image_filename]
                 if box[-1] > 0.25 and np.min(box[2:4]) > 30 and np.max(box[2:4]) > 60]

        if not boxes:
            continue

        iou_matrix = np.array([[boxlib.iou(gt_person.bbox, box[:4])
                                for box in boxes]
                               for gt_person in gt_people])
        gt_indices, box_indices = scipy.optimize.linear_sum_assignment(-iou_matrix)
        for i_gt, i_det in zip(gt_indices, box_indices):
            ex = gt_people[i_gt]
            if iou_matrix[i_gt, i_det] > 0.5 and ex.occ_fraction < 0.75:
                ex.bbox = np.array(boxes[i_det][:4])
                del ex.occ_fraction
                yolo_examples.append(ex)

    stage1_ds.examples[0] = yolo_examples
    return stage1_ds


def get_z(ex):
    camcoords = ex.camera.world_to_camera(ex.world_coords)
    return np.mean(camcoords[..., 2])


def make_composited_examples(picked_examples, i_out, output_dir, imshape):
    """Makes a composited image, and multiple examples (one for each instance),sharing that image
    but having different box, mask and pose"""

    picked_examples.sort(key=get_z, reverse=True)
    composite_image = None
    new_image_path = f'{output_dir}/{i_out:06d}.jpg'
    new_examples = []
    rng = np.random.RandomState(i_out)
    for ex in picked_examples:
        cam = ex.camera.copy()
        delta = util.random_uniform_disc(rng) * np.array([80, 20])
        cam.turn_towards(delta + np.array(cam.intrinsic_matrix[:2, 2]))
        cam.rotate(roll=rng.uniform(-np.pi / 12, np.pi / 12))

        current_image = improc.imread(ex.image_path)
        current_fgmask = rlemasklib.decode(ex.mask)

        current_image = cameravision.reproject_image_fast(current_image, ex.camera, cam, imshape)
        current_fgmask = cameravision.reproject_image_fast(
            current_fgmask, ex.camera, cam, imshape) > 0.5
        composite_image = (
            improc.blend_image(composite_image, current_image, current_fgmask)
            if composite_image is not None else current_image)
        instance_mask = rlemasklib.encode(current_fgmask.squeeze(-1))

        new_image_relpath = osp.relpath(new_image_path, DATA_ROOT)
        new_ex = ds3d.Pose3DExample(
            new_image_relpath, ex.world_coords, bbox=None, camera=cam,
            instance_mask=instance_mask)
        new_examples.append(new_ex)

    occluder_mask = rlemasklib.empty(imshape)
    for new_ex in reversed(new_examples):
        visible_mask = rlemasklib.difference(new_ex.instance_mask, occluder_mask)
        new_ex.bbox = rlemasklib.to_bbox(visible_mask)
        instance_mask_area = rlemasklib.area(new_ex.instance_mask)
        visible_mask_area = rlemasklib.area(visible_mask)
        if instance_mask_area > 0:
            new_ex.occ_fraction = 1 - visible_mask_area / instance_mask_area
        else:
            new_ex.occ_fraction = 1
        new_ex.instance_mask = visible_mask
        occluder_mask = rlemasklib.union([occluder_mask, new_ex.instance_mask])

    for new_ex in new_examples:
        new_ex.mask = occluder_mask

    spu.ensure_parent_dir_exists(new_image_path)
    imageio.imwrite(new_image_path, composite_image, quality=95)
    return new_examples


def make_combinations(examples, n_count, rng, n_people_per_image, output_dir, imshape):
    indices = np.arange(len(examples))
    composited_examples = []
    with spu.ThrottledPool() as pool:
        for i_out in spu.progressbar(range(n_count)):
            picked_indices = rng.choice(indices, n_people_per_image, replace=False)
            picked_examples = [examples[i] for i in picked_indices]
            pool.apply_async(
                make_composited_examples, (picked_examples, i_out, output_dir, imshape),
                callback=composited_examples.extend)

    return composited_examples
