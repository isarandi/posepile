#!/usr/bin/env bash
#@inproceedings{cai2022humman,
#  title={{HuMMan}: Multi-modal 4d human dataset for versatile sensing and modeling},
#  author={Cai, Zhongang and Ren, Daxuan and Zeng, Ailing and Lin, Zhengyu and Yu, Tao and Wang, Wenjia and Fan,
#          Xiangyu and Gao, Yang and Yu, Yifan and Pan, Liang and Hong, Fangzhou and Zhang, Mingyuan and
#          Loy, Chen Change and Yang, Lei and Liu, Ziwei},
#  booktitle={17th European Conference on Computer Vision, Tel Aviv, Israel, October 23--27, 2022,
#             Proceedings, Part VII},
#  pages={557--577},
#  year={2022},
#  organization={Springer}
#}
# https://caizhongang.github.io/projects/HuMMan/

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/humman"

# Download recon_kinect_color_part_{1,2,3}.zip, recon_kinect_mask.zip, recon_smpl_params.zip and recon_cameras.zip

extractrm recon_cameras.zip
extractrm recon_smpl_params.zip
extractrm recon_kinect_mask.zip

for i in {1..3}; do
  python -m posepile.ds.humman.extract_as_jpegs --zip-path="recon_kinect_color_part_$i.zip"
done

rm recon_kinect_color_part_{1,2,3}.zip