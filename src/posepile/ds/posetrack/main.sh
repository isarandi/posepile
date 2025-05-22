#!/usr/bin/env bash
# @inproceedings{andriluka2018posetrack,
#  title={Posetrack: A benchmark for human pose estimation and tracking},
#  author={Andriluka, Mykhaylo and Iqbal, Umar and Insafutdinov, Eldar and Pishchulin, Leonid and Milan, Anton and Gall, Juergen and Schiele, Bernt},
#  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
#  pages={5167--5176},
#  year={2018}
#}
# https://posetrack.net/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/posetrack"

# Batch size 1, because the image resolutions are different in this dataset and
# the detection script doesn't support batching images of different resolution (as of yet).
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/posetrack/images" --out-path="$DATA_ROOT/posetrack/yolov4_detections.pkl" --batch-size=1
python -m posepile.ds.posetrack.main