#!/usr/bin/env bash
# @article{cheng2022generalizable,
#    title={Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
#    author={Cheng, Wei and Xu, Su and Piao, Jingtan and Qian, Chen and Wu, Wayne and Lin, Kwan-Yee and Li, Hongsheng},
#    journal={arXiv preprint arXiv:2204.11798},
#    year={2022}
#}
# https://github.com/generalizable-neural-performer/gnr
# https://openxlab.org.cn/datasets/OpenXDLab/GeneBody

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=genebody
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

extractrm GeneBody.tar.gz

cd GeneBody
extractrm GeneBody_smpl_depth.tar.gz

cd GeneBody-Test10
extractrm *.tar.gz

mv GeneBody/* ./

mv GeneBody/GeneBody/* GeneBody/
rmdir GeneBody/GeneBody

# Download the GeneBody-Train40 data as well
cd "$dataset_dir/GeneBody-Train40"
extractrm *.tar.gz

python -m posepile.ds.genebody.compress_masks

python -m posepile.ds.genebody.main