import itertools

import more_itertools
import numpy as np
from addict import Addict

class JointInfo:
    def __init__(self, joints, edges=()):
        if isinstance(joints, dict):
            self.ids = joints
        elif isinstance(joints, (list, tuple, np.ndarray)):
            self.ids = JointInfo.make_id_map(joints)
        elif isinstance(joints, str):
            self.ids = JointInfo.make_id_map(joints.split(','))
        else:
            raise Exception()

        self.names = list(sorted(self.ids.keys(), key=self.ids.get))
        # the index of the joint on the opposite side (e.g. maps index of left wrist to index
        # of right wrist)
        self.mirror_mapping = [
            self.ids[JointInfo.other_side_joint_name(name)] for name in self.names]

        self.stick_figure_edges = []
        self.add_edges(edges)

    @property
    def n_joints(self):
        return len(self.ids)

    def add_edges(self, edges):
        if isinstance(edges, str):
            for path_str in edges.split(','):
                joint_names = path_str.split('-')
                for joint_name1, joint_name2 in more_itertools.pairwise(joint_names):
                    if joint_name1 in self.ids and joint_name2 in self.ids:
                        i1 = self.ids[joint_name1]
                        i2 = self.ids[joint_name2]
                        edge1 = tuple(sorted((i1, i2)))
                        edge2 = tuple(sorted((self.mirror_mapping[i1], self.mirror_mapping[i2])))
                        self.stick_figure_edges.append(edge1)
                        if edge2 != edge1:
                            self.stick_figure_edges.append(edge2)
        else:
            self.stick_figure_edges.extend(edges)

    def update_names(self, new_names):
        if isinstance(new_names, str):
            new_names = new_names.split(',')

        self.names = new_names
        new_ids = Addict()
        for i, new_name in enumerate(new_names):
            new_ids[new_name] = i
        self.ids = new_ids

    @staticmethod
    def make_id_map(names):
        return Addict(dict(zip(names, itertools.count())))

    @staticmethod
    def other_side_joint_name(name):
        if name.startswith('l'):
            return 'r' + name[1:]
        elif name.startswith('r'):
            return 'l' + name[1:]
        else:
            return name

    def select_joints(self, selected_joint_ids):
        selected_joint_ids = list(selected_joint_ids)
        new_names = [self.names[i] for i in selected_joint_ids]
        new_edges = [(selected_joint_ids.index(i), selected_joint_ids.index(j))
                     for i, j in self.stick_figure_edges
                     if i in selected_joint_ids and j in selected_joint_ids]
        return JointInfo(new_names, new_edges)

    def __str__(self):
        joint_str = ', '.join(self.names)
        edge_str = ', '.join('-'.join(self.names[i] for i in edge)
                             for edge in self.stick_figure_edges)
        return f'Joints: {joint_str}\nBones: {edge_str}'


def get_joint2bone_mat(joint_info):
    n_bones = len(joint_info.stick_figure_edges)
    joints2bones = np.zeros([n_bones, joint_info.n_joints], np.float32)
    for i_bone, (i_joint1, i_joint2) in enumerate(joint_info.stick_figure_edges):
        joints2bones[i_bone, i_joint1] = 1
        joints2bones[i_bone, i_joint2] = -1
    return joints2bones
