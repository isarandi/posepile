#!/usr/bin/env bash
#    @inproceedings{peng2021neural,
#        title={Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans},
#        author={Peng, Sida and Zhang, Yuanqing and Xu, Yinghao and Wang, Qianqian and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
#        booktitle={CVPR},
#        year={2021}
#    }
#
#
#    @inproceedings{fang2021mirrored,
#        title={Reconstructing 3D Human Pose by Watching Humans in the Mirror},
#        author={Fang, Qi and Shuai, Qing and Dong, Junting and Bao, Hujun and Zhou, Xiaowei},
#        booktitle={CVPR},
#        year={2021}
#    }
# https://chingswy.github.io/Dataset-Demo/

set -euo pipefail
source posepile/functions.sh
check_data_root

dataset_name=zjumocap
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Download all files from the Google Drive
# https://github.com/glotlabs/gdrive
gdrive files download --recursive 16GgIYBidWL5a9rjcA13oKbX22wTT5xMo
mv zjumocap-public/* ./
rmdir zjumocap-public

extractrm *.tar.gz

python -m posepile.ds.zjumocap.main