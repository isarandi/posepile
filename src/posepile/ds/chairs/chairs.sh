#!/usr/bin/env bash
# @inproceedings{jiang2023chairs,
#  title={Full-Body Articulated Human-Object Interaction},
#  author={Jiang, Nan and Liu, Tengyu and Cao, Zhexuan and Cui, Jieming and Chen, Yixin and Wang, He and Zhu, Yixin and Huang, Siyuan},
#  booktitle={ICCV},
#  year={2023}
#}
# https://yzhu.io/publication/hoi2023iccv/
# https://jnnan.github.io/project/chairs/
# https://github.com/jnnan/chairs

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/chairs"

# Download all files from the Google Drive
# https://github.com/glotlabs/gdrive
gdrive files download 1EuCwlvVcqk3CTeStf6ljNw9fNZ_A78Wv
gdrive files download 1VDCuJRBfEkHRusJvCCE50nkrHE_9Qj7X
gdrive files download 1OKwLkfacDJtdiEtOO3nLzhHudq3PmkpA