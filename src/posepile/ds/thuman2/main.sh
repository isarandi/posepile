#!/usr/bin/env bash
# @article{an2022mri,
#  title={mri: Multi-modal 3d human pose estimation dataset using mmwave, rgb-d, and inertial sensors},
#  author={An, Sizhe and Li, Yin and Ogras, Umit},
#  journal={Advances in Neural Information Processing Systems},
#  volume={35},
#  pages={27414--27426},
#  year={2022}
#}
# https://github.com/ytrock/THuman2.0-Dataset
# http://github.com/sizhean/mri
# https://zenodo.org/records/10064764

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=thuman2
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Download
# THUman2.0 Release Smpl-X Paras.zip
# THuman2.0_smpl.zip
# THuman2.0_Release_copy.zip.zip（副本）
mv *_Release_copy* THuman2.0_Release_copy.zip

extractrm *.zip
mv *Smpl-X* smplx
mv THuman2.0_smpl smpl

blender --background --python "$CODE_DIR/ds/thuman2/render.py"

python -m posepile.ds.thuman2.main