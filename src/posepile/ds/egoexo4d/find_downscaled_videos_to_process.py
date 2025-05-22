import simplepyutils as spu
import os.path as osp
from posepile.paths import DATA_ROOT

def main():
    all_videos = spu.sorted_recursive_glob(f'{DATA_ROOT}/egoexo4d/takes/**/*.mp4')
    needed_highres_videos = [
        v for v in all_videos if 'downscaled' not in v and 'aria' not in v and 'preview' not in v]
    needed_lowres_videos = [
        osp.join(osp.dirname(v), 'downscaled/448', osp.basename(v)) for v in needed_highres_videos]
    for v in needed_lowres_videos:
        print(osp.relpath(v, f'{DATA_ROOT}/egoexo4d'))

if __name__ == '__main__':
    main()