#!/usr/bin/env bash
# @inproceedings{HuCVPR2019,
#  author = {Yuan-Ting Hu and Hong-Shuo Chen and Kexin Hui and Jia-Bin Huang and Alexander G. Schwing},
#  title = { {SAIL-VOS: Semantic Amodal Instance Level Video Object Segmentation -- A Synthetic Dataset and Baselines} },
#  booktitle = {Proc. CVPR},
#  year = {2019},
#}
# https://sailvos.web.illinois.edu/_site/index.html
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/sailvos"

wget http://sailvos.web.illinois.edu/_site/assests/download_sailvos.sh
sh download_sailvos.sh
rm sailvos_*.tar.* sailvos_*.zip

# SAIL-VOS human poses are not released publicly.
# Contact SAIL-VOS maintainers if you need pose data.
extractrm sailvos_pose_training.tar sailvos_pose_val.tar
extractrm sailvos_pose_training/*.zip sailvos_pose_val/*.zip

# Do some compression on the SAILVOS data as it's very large out of the box
# Compress the SAILVOS bmp image files as they take up a huge amount of disk space
python -m posepile.sailvos.convert_to_jpeg "$DATA_ROOT/sailvos"'/**/*.bmp'

# Compress npy visibility files to npz (by a factor of ~500, from 2 MB per frame to about 4 KB)
python -m posepile.sailvos.compress_visibilities

# Compute the camera calibration matrices from the 2D and 3D human poses
# (camera matrices are not given in the dataset)
python -m posepile.sailvos.calibrate_cameras

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/sailvos" \
  --out-path="$DATA_ROOT/sailvos/yolov4_detections.pkl"

python -m posepile.ds.sailvos.main