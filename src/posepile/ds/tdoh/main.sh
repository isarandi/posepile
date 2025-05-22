#!/usr/bin/env bash
#@inproceedings{Zhang20CVPR,
#  author={Zhang, Tianshu and Huang, Buzhen and Wang, Yangang},
#  booktitle=CVPR,
#  title={Object-Occluded Human Shape and Pose Estimation From a Single Color Image},
#  year={2020},
#}
# https://www.yangangwang.com/papers/ZHANG-OOH-2020-03.html
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/3doh"

# Download trainset.zip and testset.zip
# Link: https://seueducn1-my.sharepoint.com/personal/yangangwang_seu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyangangwang%5Fseu%5Fedu%5Fcn%2FDocuments%2F3DOH50K

extractrm trainset.zip
extractrm testset.zip

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/3doh" --out-path="$DATA_ROOT/3doh/yolov4_detections.pkl"
python -m posepile.ds.tdoh.main