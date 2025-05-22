import numpy as np
import simplepyutils as spu

import posepile.datasets3d as ds3d
from posepile.joint_info import JointInfo
from posepile.paths import DATA_ROOT


def main():
    ji = ds3d.get_joint_info('huge8')
    ji_kinect = get_kinect_joint_info()
    ji_with_kinect_names = list(ji.names) + list(ji_kinect.names)
    ji_with_kinect_edges = list(ji.stick_figure_edges) + [
        (i + ji.n_joints, j + ji.n_joints)
        for i, j in ji_kinect.stick_figure_edges]

    ji = JointInfo(ji_with_kinect_names, ji_with_kinect_edges)
    new_names = [(n + ('_smpl' if '_' not in n else '')) for n in ji.names]
    ji.update_names(new_names)

    skeletons = [
        dict(
            name='smpl_24',
            suffix='smpl',
            joint_names='pelv,lhip,rhip,bell,lkne,rkne,spin,lank,rank,thor,ltoe,rtoe,neck,lcla,'
                        'rcla,head,lsho,rsho,lelb,relb,lwri,rwri,lhan,rhan',
            new_joint_names='pelv,lhip,rhip,spi1,lkne,rkne,spi2,lank,rank,spi3,ltoe,rtoe,neck,'
                            'lcla,rcla,head,lsho,rsho,lelb,relb,lwri,rwri,lhan,rhan'),
        dict(
            name='h36m_17',
            suffix='h36m',
            joint_names='pelv,rhip,rkne,rank,lhip,lkne,lank,spin,neck,head,htop,lsho,lelb,lwri,'
                        'rsho,relb,rwri'),
        dict(
            name='h36m_25',
            suffix='h36m',
            joint_names='rhip,rkne,rank,rfoo,rtoe,lhip,lkne,lank,lfoo,ltoe,spin,neck,head,htop,'
                        'lsho,lelb,lwri,lthu,lfin,rsho,relb,rwri,rthu,rfin,pelv'),
        dict(
            name='mpi_inf_3dhp_17',
            suffix='3dhp',
            joint_names='htop,neck,rsho,relb,rwri,lsho,lelb,lwri,rhip,rkne,rank,lhip,lkne,lank,'
                        'pelv,spin,head'),
        dict(
            name='mpi_inf_3dhp_28',
            suffix='3dhp',
            joint_names='thor,spi4,spi2,spin,pelv,neck,head,htop,lcla,lsho,lelb,lwri,lhan,rcla,'
                        'rsho,relb,rwri,rhan,lhip,lkne,lank,lfoo,ltoe,rhip,rkne,rank,rfoo,rtoe'),
        dict(
            name='coco_19',
            suffix='coco',
            joint_names='neck,nose,pelv,lsho,lelb,lwri,lhip,lkne,lank,rsho,relb,rwri,rhip,rkne,'
                        'rank,leye,lear,reye,rear'),
        dict(
            name='sailvos_26',
            suffix='sailvos'),
        dict(
            name='gpa_34',
            suffix='gpa'),
        dict(
            name='aspset_17',
            suffix='aspset'),
        dict(
            name='bml_movi_87',
            suffix='bmlmovi'),
        dict(
            name='mads_19',
            suffix='mads'),
        dict(
            name='berkeley_mhad_43',
            suffix='bmhad'),
        dict(
            name='total_capture_21',
            suffix='totcap'),
        dict(
            name='jta_22',
            suffix='jta'),
        dict(
            name='ikea_asm_17',
            suffix='ikea'),
        dict(
            name='human4d_32',
            suffix='human4d'),
        dict(
            name='smplx_42',
            suffix='smplx'),
        dict(
            name='ghum_35',
            suffix='ghum'),
        dict(
            name='lsp_14',
            suffix='3doh'),
        dict(
            name='3dpeople_29',
            suffix='3dpeople'),
        dict(
            name='umpm_15',
            suffix='umpm'),
        dict(
            name='kinectv2_25',
            suffix='kinectv2',
            joint_names='pelv,bell,head,htop,lsho,lelb,lwri,lhan,rsho,relb,rwri,rhan,lhip,lkne,'
                        'lank,ltoe,rhip,rkne,rank,rtoe,neck,lfin,lthu,rfin,rthu',
            new_joint_names='spinebase,spinemid,neck,head,lsho,lelb,lwri,lhan,rsho,relb,rwri,'
                            'rhan,lhip,lkne,lank,lfoo,rhip,rkne,rank,rfoo,spineshoulder,lhandtip,'
                            'lthu,rhandtip,rthu'),
    ]

    results = {}
    for skel in skeletons:
        if 'joint_names' not in skel:
            suflen = len(skel['suffix']) + 1
            skel['joint_names'] = ','.join(
                [n[:-suflen] for n in ji.names if n.endswith(skel['suffix'])])
        if 'new_joint_names' not in skel:
            skel['new_joint_names'] = skel['joint_names']

        names = skel['joint_names'].split(',')
        new_names = skel['new_joint_names'].split(',')
        assert len(new_names) == len(names)

        suffixed_names = [name + '_' + skel['suffix'] for name in names]
        try:
            indices = [ji.ids[name] for name in suffixed_names]
        except:
            print(skel['name'] + ' not working')
            continue
        ji_new = ji.select_joints(indices)
        ji_new.update_names(new_names)
        if skel['name'] == 'mpi_inf_3dhp_17':
            ji_new.add_edges('rsho-neck-spin,lsho-neck')
        results[skel['name']] = dict(
            indices=indices, names=ji_new.names, edges=ji_new.stick_figure_edges)

    ji_viz, ji_viz_indices = create_viz_joint_info(ji)
    viz_names = [n.replace('_3dhp', '_mpi_inf_3dhp') for n in ji_viz.names]
    results['smpl+head_30'] = dict(
        indices=ji_viz_indices, names=viz_names, edges=ji_viz.stick_figure_edges)
    results[''] = dict(
        indices=list(range(ji.n_joints)), names=ji.names, edges=ji.stick_figure_edges)

    spu.dump_pickle(
        results, f'{DATA_ROOT}/skeleton_conversion/skeleton_types_huge8_with_kinect.pkl')


def get_kinect_joint_info():
    names = (
        'pelv,bell,head,htop,rsho,relb,rwri,rhan,lsho,lelb,lwri,lhan,rhip,rkne,rank,rtoe,lhip,'
        'lkne,lank,ltoe,neck,rfin,rthu,lfin,lthu')
    edges = 'htop-head-neck-bell-pelv-rhip-rkne-rank-rtoe,neck-rsho-relb-rwri-rthu,rwri-rhan-rfin'
    ji = JointInfo(names, edges)
    ji.update_names([n + '_kinectv2' for n in ji.names])
    return ji


def create_viz_joint_info(base_joint_info):
    def is_smpl_keypoint(i):
        name = base_joint_info.names[i]
        return name.endswith('_smpl')

    def is_cmu_face_keypoint(i):
        name = base_joint_info.names[i]
        return any(name.endswith(f'{x}_coco') for x in 'eye nose ear'.split())

    edges_selected = [
        (j1, j2)
        for j1, j2 in base_joint_info.stick_figure_edges
        if ((is_cmu_face_keypoint(j1) and is_cmu_face_keypoint(j2))
            or (is_smpl_keypoint(j1) and is_smpl_keypoint(j2)))]

    new_joint_info = JointInfo(base_joint_info.names, edges_selected)
    new_joint_info.add_edges('htop_3dhp-head_smpl-nose_coco')

    joints_selected = sorted(set(list(np.array(new_joint_info.stick_figure_edges).reshape(-1))))
    assert 23 in joints_selected
    joints_selected = [23, *[x for x in joints_selected if x != 23]]
    return new_joint_info.select_joints(joints_selected), joints_selected
