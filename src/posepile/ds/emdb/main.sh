#!/usr/bin/env bash

# @inproceedings{kaufmann2023emdb,
#  author = {Kaufmann, Manuel and Song, Jie and Guo, Chen and Shen, Kaiyue and Jiang, Tianjian and Tang, Chengcheng and Z{\'a}rate, Juan Jos{\'e} and Hilliges, Otmar},
#  title = {{EMDB}: The {E}lectromagnetic {D}atabase of {G}lobal 3{D} {H}uman {P}ose and {S}hape in the {W}ild},
#  booktitle = {International Conference on Computer Vision (ICCV)},
#  year = {2023}
#}
# https://eth-ait.github.io/emdb/

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/emdb"

# Download the zip files

for name in *.zip; do
  extractrm "$name"
done