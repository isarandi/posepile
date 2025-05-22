#!/usr/bin/env bash
#@inproceedings{tripathi2023ipman,
#    title = {{3D} Human Pose Estimation via Intuitive Physics},
#    author = {Tripathi, Shashank and M{\"u}ller, Lea and Huang, Chun-Hao P. and Taheri Omid
#    and Black, Michael J. and Tzionas, Dimitrios},
#    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
#    Recognition (CVPR)},
#    month = {June},
#    year = {2023}
#}
# https://moyo.is.tue.mpg.de

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/moyo"

git clone https://github.com/sha2nkt/moyo_toolkit.git


python -m posepile.ds.moyo.main --stage=1

# find "$DATA_ROOT/moyo_downscaled/" -name '*.jpg' | wc -l
# should be around 244294

for i in {1..24}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/moyo_downscaled" --out-dir="$DATA_ROOT/moyo_downscaled/masks"
done

python -m posepile.ds.moyo.main --stage=2

python -m posepile.ds.moyo.to_barecat

