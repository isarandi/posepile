#!/usr/bin/env bash
# @InProceedings{Fieraru_2021_CVPR,
# author = {Fieraru, Mihai and Zanfir, Mihai and Pirlea, Silviu-Cristian and Olaru, Vlad and Sminchisescu, Cristian},
# title = {AIFit: Automatic 3D Human-Interpretable Feedback Models for Fitness Training},
# booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
# year = {2021}}
# https://fit3d.imar.ro/download

# @inproceedings{fieraru2021learning,
# title={Learning complex 3d human self-contact},
# author={Fieraru, Mihai and Zanfir, Mihai and Oneata, Elisabeta and Popa, Alin-Ionut and Olaru, Vlad and Sminchisescu, Cristian},
# booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
# volume={35},
# number={2},
# pages={1343--1351},
# year={2021} }
# https://sc3d.imar.ro/humansc3d

# @InProceedings{Fieraru_2020_CVPR,
# author = {Fieraru, Mihai and Zanfir, Mihai and Oneata, Elisabeta and Popa, Alin-Ionut and Olaru, Vlad and Sminchisescu, Cristian},
# title = {Three-Dimensional Reconstruction of Human Interactions},
# booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
# month = {June},
# year = {2020} }
# https://ci3d.imar.ro/chi3d
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/imar_datasets"

mkdir fit3d humansc3d chi3d

# Download data (.tar.gz and .json files) from the websites of CHI3D, Fit3D and HumanSC3D
# into above subdirs

extract_and_delete() {
  pushd "$(dirname "$1")"
  tar xf "$1"
  rm "$1"
  popd
}

for name in fit3d humansc3d chi3d; do
  extract_and_delete "${name}_train.tar.gz"
  extract_and_delete "${name}_test.tar.gz"
done

for name in fit3d humansc3d chi3d; do
  python -m posepile.ds.imar.main --dataset="$name" --stage=1
done

# Approx. number of samples selected
# Fit3D: 147314
# HumanSC3D: 71847
# CHI3D: 46231

segment_dataset() {
  dataset_name=$1
  max_task_id=$2
  for i in $(seq 0 "$max_task_id"); do
    SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/${dataset_name}_downscaled" --out-dir="$DATA_ROOT/${dataset_name}_downscaled/masks"
  done
}

segment_dataset fit3d 14
segment_dataset humansc3d 7
segment_dataset chi3d 4

for name in fit3d humansc3d chi3d; do
  python -m posepile.ds.imar.main --dataset="$name" --stage=2
done