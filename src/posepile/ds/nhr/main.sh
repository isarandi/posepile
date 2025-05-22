#!/usr/bin/env bash
#@inproceedings{wu2020multi,
#   title={Multi-View Neural Human Rendering},
#   author={Wu, Minye and Wang, Yuehao and Hu, Qiang and Yu, Jingyi},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={1682--1691},
#   year={2020}
#}
#https://wuminye.github.io/NHR/

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/nhr"
