import argparse

import numpy as np
import simplepyutils as spu
import simplepyutils as spu
from simplepyutils import FLAGS

import posepile.compositing
import posepile.ds.surreal.main
from posepile.paths import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int)
    spu.initialize(parser)

    if FLAGS.stage == 1:
        make_stage1()
    elif FLAGS.stage == 2:
        make_dataset()


@spu.picklecache('muco_surreal2_stage1.pkl', min_time="2020-07-18T00:43:46")
def make_stage1():
    ds = posepile.ds.surreal.main.make_dataset()
    ds.examples[0] = posepile.compositing.make_combinations(
        ds.examples[0], n_count=len(ds.examples[0]) // 3, rng=np.random.RandomState(0),
        n_people_per_image=4, output_dir=f'{DATA_ROOT}/muco-surreal2/images',
        imshape=(240, 320))
    return ds


@spu.picklecache('muco_surreal2.pkl', min_time="2020-07-18T00:43:46")
def make_dataset():
    muco_surreal = make_stage1()
    all_boxes = spu.load_pickle(f'{DATA_ROOT}/muco-surreal2/yolov4_detections.pkl')
    return posepile.compositing.make_composited_dataset(muco_surreal, all_boxes)


if __name__ == '__main__':
    main()
