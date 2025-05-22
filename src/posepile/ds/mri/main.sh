#!/usr/bin/env bash
# @article{an2022mri,
#  title={mri: Multi-modal 3d human pose estimation dataset using mmwave, rgb-d, and inertial sensors},
#  author={An, Sizhe and Li, Yin and Ogras, Umit},
#  journal={Advances in Neural Information Processing Systems},
#  volume={35},
#  pages={27414--27426},
#  year={2022}
#}
# https://datadryad.org/stash/dataset/doi:10.5061/dryad.9ghx3ffpp
# https://sizhean.github.io/mri
# http://github.com/sizhean/mri
# https://zenodo.org/records/10064764

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=mri
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

wget https://datadryad.org/stash/downloads/file_stream/2738861 -O blurred_videos.zip
wget https://datadryad.org/stash/downloads/file_stream/2738858 -O dataset_release.zip
wget https://datadryad.org/stash/downloads/file_stream/2738863 -O README.md
extractrm *.zip

python -m humcentr_cli.detect_people_video --video-dir="$dataset_dir" --output-dir="$dataset_dir/detections" --file-pattern='**/*.mp4'

python -m posepile.ds.mri.main