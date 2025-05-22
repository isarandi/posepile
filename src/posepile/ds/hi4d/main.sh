#!/usr/bin/env bash
#@inproceedings{yin2023hi4d,
#      author = {Yin, Yifei and Guo, Chen and Kaufmann, Manuel and Zarate, Juan and Song, Jie and Hilliges, Otmar},
#      title = {Hi4D: 4D Instance Segmentation of Close Human Interaction},
#      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
#      year = {2023}
#      }
# https://yifeiyin04.github.io/Hi4D/

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=hi4d
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

token=your-token-here

for n in 00_1 00_2 01 02_1 02_2 09 10 12 13_1 13_2 14 15_1 15_2 16 17_1 17_2 18_1 18_2 19 21_1 21_2 22 23_1 23_2 27_1 27_2 28 32 37_1 37_2; do
    wget "https://hi4d.ait.ethz.ch/download.php?dt=${token}&file=/pair${n}.tar.gz" -O "pair${n}.tar.gz"
done


python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/hi4d" --out-path="$DATA_ROOT/hi4d/yolov4_detections.pkl"
python -m posepile.ds.hi4d.main

# render additional images with Blender



python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/hi4d_rerender" --out-path="$DATA_ROOT/hi4d_rerender/yolov4_detections.pkl" --batch-size=8