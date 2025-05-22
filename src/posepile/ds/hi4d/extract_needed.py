import tarfile

import simplepyutils as spu
from posepile.paths import DATA_ROOT


def main():
    archive_paths = spu.sorted_recursive_glob(f'{DATA_ROOT}/hi4d/*.tar.gz')
    exclude_substrings = ['/frames', '/frames_vis', '/seg/instance', '/seg/mesh_seg_mask']
    for archive_path in archive_paths:
        with tarfile.open(archive_path) as tarf:
            for pbar, member in spu.zip_progressbar(tarf):
                #if not any(s in member.name for s in exclude_substrings):
                #if 'img_seg_mask' in member.name:
                if '/frames/' in member.name:
                    pbar.set_description(member.name)
                    tarf.extract(member, path=f'{DATA_ROOT}/hi4d/')


if __name__ == '__main__':
    main()
