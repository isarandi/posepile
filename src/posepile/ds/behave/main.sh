#!/usr/bin/env bash
#  @inproceedings{bhatnagar22behave,
#  title = {BEHAVE: Dataset and Method for Tracking Human Object Interactions},
#  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
#  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
#  year = {2022},
#  }
# https://virtualhumans.mpi-inf.mpg.de/behave/
set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=behave
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

for i in {01..07}; do
  wget "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date${i}.zip"
done

wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/calibs.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/split.json

unzip Date*.zip -d sequences
rm Date*.zip
extractrm calibs.zip

python -m humcentr_cli.detect_people --image-root="$dataset_name" --out-path="$dataset_name/yolov4_detections.pkl" --file-pattern='**/*.color.jpg'

python -m posepile.ds.behave.main
