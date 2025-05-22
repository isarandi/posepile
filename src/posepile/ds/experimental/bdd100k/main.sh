#!/usr/bin/env bash
#@InProceedings{bdd100k,
#    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
#              Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
#    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
#    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month = {June},
#    year = {2020}
#}
# https://arxiv.org/abs/1805.04687
# https://github.com/bdd100k/bdd100k
# https://bdd-data.berkeley.edu/
# https://www.vis.xyz/bdd100k/
# https://doc.bdd100k.com/download.html

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/bdd100k"

extractrm bdd100k_pose_labels_trainval.zip
mv bdd100k_pose_labels_trainval/bdd100k/labels ./
rmdir -p bdd100k_pose_labels_trainval/bdd100k