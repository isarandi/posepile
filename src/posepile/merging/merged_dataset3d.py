import argparse
import random

import simplepyutils as spu
from addict import Addict
from simplepyutils import FLAGS

import posepile.datasets3d as ds3d
from posepile import joint_filtering
from posepile.joint_info import JointInfo
from posepile.util import TEST, TRAIN, VALID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    spu.initialize(parser)

    if FLAGS.name == 'huge8':
        make_huge8()
    elif FLAGS.name == 'small5':
        make_small5()
    elif FLAGS.name == 'medium3':
        make_medium3()
    else:
        raise ValueError()


def merge_datasets(datasets_with_uses_suf):
    merged_joint_info = merge_joint_infos_of_datasets(datasets_with_uses_suf)

    for i_ds in range(len(datasets_with_uses_suf)):
        datasets_with_uses_suf[i_ds][0] = joint_filtering.convert_sparse_dataset(
            datasets_with_uses_suf[i_ds][0], merged_joint_info, update_bones=False),

    examples = {TRAIN: [], VALID: [], TEST: []}
    for ds, uses, suf in datasets_with_uses_suf:
        for take, use_as in uses:
            examples[use_as] += ds.examples[take]

    return ds3d.Pose3DDataset(
        merged_joint_info, examples[0], examples[1], examples[2], compute_bone_lengths=False)


def merge_joint_infos_of_datasets(datasets_with_uses_suf):
    datasets = [ds for ds, uses, suf in datasets_with_uses_suf]
    suffixes = [suf for ds, uses, suf in datasets_with_uses_suf]
    ds_done = set()
    for ds, suffix in zip(datasets, suffixes):
        if ds in ds_done:
            continue
        suffix = '' if suffix == '' else '_' + suffix
        ds.joint_info = convert_joint_info(ds.joint_info, suffix, [])
        ds_done.add(ds)

    joint_infos = [ds.joint_info for ds in datasets]
    all_joint_names = [n for ji in joint_infos for n in ji.names]
    merged_ids = JointInfo.make_id_map(get_unique_elements(all_joint_names))
    print(merged_ids)
    edge_names = [[ji.names[j1], ji.names[j2]]
                  for ji in joint_infos for j1, j2 in ji.stick_figure_edges]
    print(edge_names)
    edge_ids = [(merged_ids[e1], merged_ids[e2]) for e1, e2 in edge_names]
    print(edge_ids)
    edge_ids = sorted(set(tuple(sorted((j1, j2))) for j1, j2 in edge_ids))
    print(edge_ids)
    merged_joint_info = JointInfo(merged_ids, edge_ids)
    return merged_joint_info


def merge_joint_infos(joint_infos_with_suf):
    joint_infos = [ji for ji, suf in joint_infos_with_suf]
    suffixes = [suf for ds, suf in joint_infos_with_suf]
    joint_infos_new = []
    ji_done = set()
    for ji, suffix in zip(joint_infos, suffixes):
        if ji in ji_done:
            continue
        suffix = '' if suffix == '' else '_' + suffix
        ji_done.add(ji)
        ji_new = convert_joint_info(ji, suffix, [])
        joint_infos_new.append(ji_new)

    all_joint_names = [n for ji in joint_infos_new for n in ji.names]
    merged_ids = ds3d.JointInfo.make_id_map(get_unique_elements(all_joint_names))
    print(merged_ids)
    edge_names = [[ji.names[j1], ji.names[j2]]
                  for ji in joint_infos_new for j1, j2 in ji.stick_figure_edges]
    print(edge_names)
    edge_ids = [(merged_ids[e1], merged_ids[e2]) for e1, e2 in edge_names]
    print(edge_ids)
    edge_ids = sorted(set(tuple(sorted((j1, j2))) for j1, j2 in edge_ids))
    print(edge_ids)
    merged_joint_info = ds3d.JointInfo(merged_ids, edge_ids)
    return merged_joint_info


@spu.picklecache('huge8.pkl', min_time="2022-08-07T01:41:28")
def make_huge8():
    d = Addict()
    d.h36m = ds3d.get_compressed_dataset('h36m_alljoints')
    d.h36m.update_bones()
    d.fit3d = ds3d.get_compressed_dataset('imar', 'fit3d')
    d.humansc3d = ds3d.get_compressed_dataset('imar', 'humansc3d')
    d.chi3d = ds3d.get_compressed_dataset('imar', 'chi3d')
    d.muco_3dhp = ds3d.get_compressed_dataset('muco_3dhp_200k')

    d.humbi = ds3d.get_dataset('humbi')
    humbi_coco_joint_ids = [
        i for i, name in enumerate(d.humbi.joint_info.names) if name.endswith('_coco')]
    new_humbi_joint_info = d.humbi.joint_info.select_joints(humbi_coco_joint_ids)
    d.humbi = data.joint_filtering.convert_dataset(d.humbi, new_humbi_joint_info)
    ds3d.compress_dataset(d.humbi)
    print('HUMBI loaded')

    import posepile.ds.rich.main
    d.rich = posepile.ds.rich.main.make_rich_body()
    ds3d.compress_dataset(d.rich)
    print('RICH loaded')

    import posepile.ds.tdhp.full
    d.tdhp_full = posepile.ds.tdhp.full.make_dataset()
    ds3d.compress_dataset(d.tdhp_full)
    print('3DHP-full loaded')

    d.tdhp = ds3d.get_dataset('tdhp')
    d.tdhp.examples[0].clear()
    d.tdhp.examples[1].clear()
    d.tdhp = joint_filtering.convert_dataset(d.tdhp, d.muco_3dhp.joint_info, update_bones=False)
    ds3d.compress_dataset(d.tdhp)
    print('3DHP loaded')

    d.mupots = ds3d.get_dataset('mupots_yolo')
    d.mupots = joint_filtering.convert_dataset(d.mupots, d.muco_3dhp.joint_info, update_bones=False)
    ds3d.compress_dataset(d.mupots)
    print('MuPoTS loaded')

    for name in ('surreal sailvos panoptic aist gpa aspset agora tdoh tdpeople bml_movi mads umpm '
                 'totalcapture bmhad jta ikea human4d behave spec rich hspace'.split()):
        d[name] = ds3d.get_compressed_dataset(name)
        print(name, 'loaded')

    print('Merging...')

    ds = merge_datasets([
        [d.surreal, [(0, 0), (1, 0)], ''],
        [d.h36m, [(0, 0), (1, 0)], 'h36m'],
        [d.fit3d, [(0, 0)], 'h36m'],
        [d.chi3d, [(0, 0)], 'h36m'],
        [d.humansc3d, [(0, 0)], 'h36m'],
        [d.sailvos, [(0, 0), (1, 0)], 'sailvos'],
        [d.panoptic, [(0, 0)], 'coco'],
        [d.muco_3dhp, [(0, 0)], '3dhp'],
        [d.aist, [(0, 0), (1, 0)], 'coco'],
        [d.gpa, [(0, 0)], 'gpa'],
        [d.humbi, [(0, 0)], ''],
        [d.aspset, [(0, 0), (1, 0)], 'aspset'],
        [d.agora, [(0, 0), (1, 0)], ''],
        [d.tdoh, [(0, 0)], '3doh'],
        [d.tdpeople, [(0, 0)], '3dpeople'],
        [d.bml_movi, [(0, 0)], 'bmlmovi'],
        [d.mads, [(0, 0)], 'mads'],
        [d.umpm, [(0, 0)], 'umpm'],
        [d.bmhad, [(0, 0)], 'bmhad'],
        [d.tdhp_full, [(0, 0)], '3dhp'],
        [d.totalcapture, [(0, 0)], 'totcap'],
        [d.jta, [(0, 0), (1, 0)], 'jta'],
        [d.ikea, [(0, 0)], 'ikea'],
        [d.human4d, [(0, 0)], 'human4d'],
        [d.behave, [(0, 0)], ''],
        [d.rich, [(0, 0), (1, 0)], 'smplx'],
        [d.spec, [(0, 0)], ''],
        [d.hspace, [(0, 0)], 'ghum'],
        [d.tdhp, [(2, 1)], '3dhp'],
        [d.mupots, [(1, 1)], '3dhp']])
    print('Filtering...')
    ds3d.filter_dataset_by_plausibility(ds, piano1=True)
    print('Training set size:', len(ds.examples[0]))
    return ds


def make_huge8_joint_info():
    jis = Addict()
    jis.h36m = ds3d.get_joint_info('h36m_alljoints')
    jis.fit3d = ds3d.get_joint_info('imar', 'fit3d')
    jis.humansc3d = ds3d.get_joint_info('imar', 'humansc3d')
    jis.chi3d = ds3d.get_joint_info('imar', 'chi3d')

    for name in ('surreal sailvos panoptic aist gpa aspset tdoh tdpeople bml_movi mads umpm '
                 'totalcapture bmhad jta ikea human4d behave spec hspace'.split()):
        jis[name] = ds3d.get_joint_info(name)

    jis.muco_3dhp = make_muco_3dhp_200k().joint_info

    jis.humbi = ds3d.get_joint_info('humbi')
    humbi_coco_joint_ids = [i for i, name in enumerate(jis.humbi.names) if name.endswith('_coco')]
    jis.humbi = jis.humbi.select_joints(humbi_coco_joint_ids)

    import posepile.ds.tdhp.full
    jis.tdhp_full = posepile.ds.tdhp.full.make_dataset().joint_info

    import posepile.ds.rich.main
    jis.rich = posepile.ds.rich.main.make_rich_body().joint_info

    jis.tdhp = jis.muco_3dhp
    jis.mupots = jis.muco_3dhp

    ds = merge_joint_infos([
        [jis.surreal, ''],  #
        [jis.h36m,  'h36m'],
        [jis.fit3d, 'h36m'],
        [jis.chi3d,  'h36m'],
        [jis.humansc3d, 'h36m'],
        [jis.sailvos,  'sailvos'],
        [jis.panoptic, 'coco'],
        [jis.muco_3dhp,  '3dhp'],
        [jis.aist, 'coco'],
        [jis.gpa,  'gpa'],
        [jis.humbi,  ''],
        [jis.aspset, 'aspset'],
        [jis.agora,  ''],  #
        [jis.tdoh, '3doh'],
        [jis.tdpeople,  '3dpeople'],
        [jis.bml_movi, 'bmlmovi'],
        [jis.mads, 'mads'],
        [jis.umpm, 'umpm'],
        [jis.bmhad, 'bmhad'],
        [jis.tdhp_full, '3dhp'],
        [jis.totalcapture,  'totcap'],
        [jis.jta,  'jta'],
        [jis.ikea, 'ikea'],
        [jis.human4d, 'human4d'],
        [jis.behave,  ''],  #
        [jis.rich,  'smplx'],  #
        [jis.spec, ''],  #
        [jis.hspace,  'ghum'],
        [jis.tdhp, '3dhp'],
        [jis.mupots, '3dhp']])
    return ds


@spu.picklecache('huge8_dummy.pkl', min_time="2022-08-07T01:41:28")
def make_huge8_dummy():
    ds = ds3d.get_dataset('huge8')
    ds.examples[0] = random.sample(ds.examples[0], 10000)
    return ds


@spu.picklecache('medium3.pkl', min_time="2022-08-18T15:20:06")
def make_medium3():
    d = Addict()
    d.h36m = ds3d.get_compressed_dataset('h36m_alljoints')
    d.h36m.update_bones()
    d.muco_3dhp = ds3d.get_compressed_dataset('muco_3dhp_200k')

    d.humbi = ds3d.get_dataset('humbi')
    humbi_coco_joint_ids = [
        i for i, name in enumerate(d.humbi.joint_info.names) if name.endswith('_coco')]
    new_humbi_joint_info = d.humbi.joint_info.select_joints(humbi_coco_joint_ids)
    d.humbi = data.joint_filtering.convert_dataset(d.humbi, new_humbi_joint_info)
    ds3d.compress_dataset(d.humbi)
    print('HUMBI loaded')

    import posepile.ds.rich.main
    d.rich = posepile.ds.rich.main.make_rich_body()
    ds3d.compress_dataset(d.rich)
    print('RICH loaded')

    import posepile.ds.tdhp.full
    d.tdhp_full = posepile.ds.tdhp.full.make_dataset()
    ds3d.compress_dataset(d.tdhp_full)
    print('3DHP-full loaded')

    d.tdhp = ds3d.get_dataset('tdhp')
    d.tdhp.examples[0].clear()
    d.tdhp.examples[1].clear()
    d.tdhp = joint_filtering.convert_dataset(d.tdhp, d.muco_3dhp.joint_info, update_bones=False)
    ds3d.compress_dataset(d.tdhp)
    print('3DHP loaded')

    d.mupots = ds3d.get_dataset('mupots_yolo')
    d.mupots = joint_filtering.convert_dataset(d.mupots, d.muco_3dhp.joint_info, update_bones=False)
    ds3d.compress_dataset(d.mupots)
    print('MuPoTS loaded')

    for name in ('surreal sailvos panoptic aist agora tdpeople '
                 'totalcapture bmhad jta rich hspace'.split()):
        d[name] = ds3d.get_compressed_dataset(name)
        print(name, 'loaded')

    print('Merging...')

    ds = merge_datasets([
        [d.surreal, [(0, 0), (1, 0)], ''],
        [d.h36m, [(0, 0), (1, 0)], 'h36m'],
        [d.sailvos, [(0, 0), (1, 0)], 'sailvos'],
        [d.panoptic, [(0, 0)], 'coco'],
        [d.muco_3dhp, [(0, 0)], '3dhp'],
        [d.aist, [(0, 0), (1, 0)], 'coco'],
        [d.humbi, [(0, 0)], ''],
        [d.agora, [(0, 0), (1, 0)], ''],
        [d.jta, [(0, 0), (1, 0)], 'jta'],
        [d.tdhp_full, [(0, 0)], '3dhp'],
        [d.tdpeople, [(0, 0)], '3dpeople'],
        [d.totalcapture, [(0, 0)], 'totcap'],
        [d.rich, [(0, 0), (1, 0)], 'smplx'],
        [d.hspace, [(0, 0)], 'ghum'],
        [d.tdhp, [(2, 1)], '3dhp'],
        [d.mupots, [(1, 1)], '3dhp']])
    print('Filtering...')
    ds3d.filter_dataset_by_plausibility(ds, piano1=True)
    print('Training set size:', len(ds.examples[0]))
    return ds


@spu.picklecache('small5.pkl', min_time="2022-07-25T12:28:40")
def make_small5():
    d = Addict()
    d.h36m = ds3d.get_compressed_dataset('h36m_alljoints')
    d.h36m.update_bones()
    d.muco_3dhp = ds3d.get_compressed_dataset('muco_3dhp_200k')

    d.tdhp = ds3d.get_dataset('tdhp')
    d.tdhp.examples[0].clear()
    d.tdhp.examples[1].clear()
    d.tdhp = joint_filtering.convert_dataset(d.tdhp, d.muco_3dhp.joint_info, update_bones=False)
    ds3d.compress_dataset(d.tdhp)

    d.mupots = ds3d.get_dataset('mupots_yolo')
    d.mupots = joint_filtering.convert_dataset(d.mupots, d.muco_3dhp.joint_info, update_bones=False)
    ds3d.compress_dataset(d.mupots)

    d.surreal = ds3d.get_compressed_dataset('surreal')

    print('Merging...')
    ds = merge_datasets([
        [d.surreal, [(0, 0), (1, 0)], ''],
        [d.h36m, [(0, 0), (1, 0)], 'h36m'],
        [d.muco_3dhp, [(0, 0)], '3dhp'],
        [d.tdhp, [(2, 1)], '3dhp'],
        [d.mupots, [(1, 1)], '3dhp']])
    print('Training set size:', len(ds.examples[0]))
    ds.update_bones()
    return ds


def convert_joint_info(joint_info, suffix, exceptions):
    ids = ds3d.JointInfo.make_id_map(
        [x + ('' if x in exceptions else suffix) for x in joint_info.names])
    return ds3d.JointInfo(ids, joint_info.stick_figure_edges)


def get_unique_elements(lst):
    uniques = []
    for x in lst:
        if x not in uniques:
            uniques.append(x)
    return uniques


if __name__ == '__main__':
    main()
