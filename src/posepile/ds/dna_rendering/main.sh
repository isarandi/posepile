#!/usr/bin/env bash
#@article{2023dnarendering,
#      title={DNA-Rendering: A Diverse Neural Actor Repository for High-Fidelity Human-centric Rendering},
#      author={Wei Cheng and Ruixiang Chen and Wanqi Yin and Siming Fan and Keyu Chen and Honglin He and Huiwen Luo and Zhongang Cai and Jingbo Wang and Yang Gao and Zhengming Yu and Zhengyu Lin and Daxuan Ren and Lei Yang and Ziwei Liu and Chen Change Loy and Chen Qian and Wayne Wu and Dahua Lin and Bo Dai and Kwan-Yee Lin},
#      journal   = {arXiv preprint},
#      volume    = {arXiv:2307.10173},
#      year    = {2023}
#}
# https://dna-rendering.github.io/
# https://github.com/DNA-Rendering/DNA-Rendering

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=dna_rendering
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Download all files from the Google Drive. "$DATA_ROOT/dna_rendering/Part 1" and "$DATA_ROOT/dna_rendering/Part 2" should exist
# https://github.com/glotlabs/gdrive
gdrive files download --recursive 1nbqxTIsTnfhz7DrzhaFE4cQ8TQvOD71T
mv "Part 1/*" ./
mv "Part 2/*" ./

extractrm *.zip

python -m posepile.ds.dna_rendering.main --stage=1
python -m posepile.ds.dna_rendering.main --stage=2