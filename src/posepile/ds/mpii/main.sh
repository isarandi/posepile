#!/usr/bin/env bash
# 2D Human Pose Estimation: New Benchmark and State of the Art Analysis.
# Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler and Bernt Schiele.
# CVPR 2014
# http://human-pose.mpi-inf.mpg.de/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/mpii"

wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
extractrm mpii_human_pose_v1.tar.gz

wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip
unzip -j mpii_human_pose_v1_u12_2.zip
rm mpii_human_pose_v1_u12_2.zip

# Originally detection was run with the https://github.com/isarandi/darknet repo as follows:
# darknet/run_yolo.sh --image-root "$DATA_ROOT/mpii" --out-path "$DATA_ROOT/mpii/yolov3_detections.pkl"
# The newer version is:
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/mpii" --out-path="$DATA_ROOT/mpii/yolov4_detections.pkl"