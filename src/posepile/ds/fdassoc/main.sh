#!/usr/bin/env bash
# @InProceedings{20204DAssociation,
#  author = {Zhang, Yuxiang and An, Liang and Yu, Tao and Li, xiu and Li, Kun and Liu, Yebin},
#  title = {4D Association Graph for Realtime Multi-person Motion Capture Using Multiple Video Cameras},
#  booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition, (CVPR)},
#  year={2020},
#}
#
#@inproceedings{lightcap2021,
# title={Light-weight Multi-person Total Capture Using Sparse Multi-view Cameras},
# author={Zhang, Yuxiang and Li, Zhe and An, Liang and Li, Mengcheng and Yu, Tao and Liu, Yebin},
# year={2021},
# booktitle={IEEE International Conference on Computer Vision}
#}
# https://github.com/zhangyux15/multiview_human_dataset

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/fdassoc"


wget https://cloud.tsinghua.edu.cn/f/82b5512334344a4187ff/?dl=1 -O markless_multiview_data.zip
wget https://cloud.tsinghua.edu.cn/f/b8f1b0729fc142b2ad87/?dl=1 -O data2.zip

for f in *.zip; do
  extractrm "$f"
done

# Additional data: https://pan.baidu.com/s/1AZgNV4kp7PuIBicEiSPdGA extract password: 05or