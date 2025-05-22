#!/usr/bin/env bash

# @inproceedings{Shapy:CVPR:2022,
  #  title = {Accurate 3D Body Shape Regression using Metric and Semantic Attributes},
  #  author = {Choutas, Vasileios and M{\"u}ller, Lea and Huang, Chun-Hao P. and Tang, Siyu and Tzionas, Dimitrios and Black, Michael J.},
  #  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  #  year = {2022}
  #}

set -euo pipefail

source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/hbw"

# Download HBW.zip

extractrm HBW.zip
