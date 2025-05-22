import warnings

import numpy as np

from posepile.util import geom3d


class AdaptivePoseSampler:
    def __init__(self, thresh, check_validity=False, assume_nan_unchanged=False):
        self.prev_pose = None
        self.thresh = thresh
        self.check_validity = check_validity
        self.assume_nan_unchanged = assume_nan_unchanged

    def should_skip(self, pose):
        pose = np.array(pose)
        if self.prev_pose is None:
            self.prev_pose = pose.copy()
            return not np.any(geom3d.are_joints_valid(pose))

        if self.check_validity:
            valid_now = geom3d.are_joints_valid(pose)
            valid_prev = geom3d.are_joints_valid(self.prev_pose)
            any_newly_valid = np.any(np.logical_and(np.logical_not(valid_prev), valid_now))
            if any_newly_valid:
                self.update(pose)
                return False
        else:
            valid_now = slice(None)

        change = np.linalg.norm(pose[valid_now] - self.prev_pose[valid_now], axis=-1)
        #  print(change)

        if self.assume_nan_unchanged:
            some_changed = np.any(change >= self.thresh)
        else:
            some_changed = not np.all(change < self.thresh)
        if some_changed:
            self.update(pose)
            return False
        return True

    def update(self, pose):
        if self.assume_nan_unchanged:
            isnan = np.isnan(pose)
            self.prev_pose[~isnan] = pose[~isnan]
        else:
            self.prev_pose[:] = pose


class AdaptivePoseSampler2:
    def __init__(self, thresh, check_validity=False, assume_nan_unchanged=False, buffer_size=1):
        self.prev_poses = RingBufferArray(buffer_size, copy_last_if_nan=assume_nan_unchanged)
        self.thresh = thresh
        self.check_validity = check_validity
        self.assume_nan_unchanged = assume_nan_unchanged

    def should_skip(self, pose):
        pose = np.array(pose)
        if self.prev_poses.array is None:
            self.prev_poses.add(pose)
            return not np.any(geom3d.are_joints_valid(pose))

        if self.check_validity:
            valid_now = geom3d.are_joints_valid(pose)
            valid_prev = geom3d.are_joints_valid(self.prev_poses.last_item())
            any_newly_valid = np.any(np.logical_and(np.logical_not(valid_prev), valid_now))
            if any_newly_valid:
                self.prev_poses.add(pose)
                return False
        else:
            valid_now = slice(None)

        change = np.linalg.norm(pose[valid_now] - self.prev_poses.array[:, valid_now], axis=-1)

        if self.assume_nan_unchanged:
            if change.size == 0:
                some_changed = False
            else:
                with np.errstate(invalid='ignore'), warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'All-NaN slice encountered')
                    minmax_change = np.nanmin(np.nanmax(change, axis=1), axis=0)
                some_changed = minmax_change >= self.thresh
        else:
            some_changed = not np.any(np.all(change < self.thresh, axis=1), axis=0)

        if some_changed:
            self.prev_poses.add(pose)
            return False
        return True


class RingBufferArray:
    def __init__(self, buffer_size, copy_last_if_nan=False):
        self.buffer_size = buffer_size
        self.array = None
        self.i_buf = 0
        self.copy_last_if_nan = copy_last_if_nan

    def add(self, item):
        if self.array is None:
            self.array = np.full(
                shape=[self.buffer_size, *item.shape], fill_value=np.nan, dtype=np.float32)

        if self.copy_last_if_nan:
            self.array[self.i_buf] = self.last_item()
            isnan = np.isnan(item)
            self.array[self.i_buf][~isnan] = item[~isnan]
        else:
            self.array[self.i_buf] = item

        self.i_buf = (self.i_buf + 1) % self.buffer_size

    def last_item(self):
        i = (self.i_buf - 1) % self.buffer_size
        return self.array[i]
