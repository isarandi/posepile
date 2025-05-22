#!/usr/bin/env bash
#@inproceedings{han2023high,
#  title={High-fidelity 3D Human Digitization from Single 2K Resolution Images},
#  author={Han, Sang-Hun and Park, Min-Gyu and Yoon, Ju Hong and Kang, Ju-Mi and Park, Young-Jae and Jeon, Hae-Gon},
#  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#  year={2023}
#}
# https://sanghunhan92.github.io/conference/2K2K/
# https://github.com/ketiVision/2K2K
# https://github.com/SangHunHan92/2K2K


set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=tk2k
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

GOOGLE_DRIVE_ID=what-you-get-via-email
gdrive files download --recursive $GOOGLE_DRIVE_ID
# Also download keypoints.zip and smplx.zip from Dropbox
# https://www.dropbox.com/scl/fi/ic3gpnz5ngr4lxquhyhyn/smplx.zip?rlkey=m1u2mfe0uvq5rz08si04jm0af&dl=0
# https://www.dropbox.com/scl/fi/6qb3a70nxjtug0lcxtrhc/keypoints.zip?rlkey=1f6rnacpfnipl3b0p9wx2z09v&dl=0

extractrm keypoints.zip smplx.zip

cd 230831/train/1M
extractrm 1M.zip

cd "$dataset_dir/230831/test"
extractrm 100K.zip
