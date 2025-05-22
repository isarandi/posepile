#!/usr/bin/env bash
# @article{HALILAJ2021110650,
#title = {American society of biomechanics early career achievement award 2020: Toward portable and modular biomechanics labs: How video and IMU fusion will change gait analysis},
#journal = {Journal of Biomechanics},
#volume = {129},
#pages = {110650},
#year = {2021},
#issn = {0021-9290},
#doi = {https://doi.org/10.1016/j.jbiomech.2021.110650},
#url = {https://www.sciencedirect.com/science/article/pii/S002192902100419X},
#author = {Eni Halilaj and Soyong Shin and Eric Rapp and Donglai Xiang},
#}
# https://simtk.org/projects/cmupanopticdata
# https://github.com/CMU-MBL/CMU_PanopticDataset_2.0

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=panoptic2
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"