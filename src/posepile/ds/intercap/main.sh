#!/usr/bin/env bash
# @inproceedings{huang2022intercap,
#    title        = {{InterCap}: {J}oint Markerless {3D} Tracking of Humans and Objects in Interaction},
#    author       = {Huang, Yinghao and Taheri, Omid and Black, Michael J. and Tzionas, Dimitrios},
#    booktitle    = {{German Conference on Pattern Recognition (GCPR)}},
#    volume       = {13485},
#    pages        = {281--299},
#    year         = {2022},
#    organization = {Springer},
#    series       = {Lecture Notes in Computer Science}
#}
# https://intercap.is.tue.mpg.de/index.html
# https://github.com/YinghaoHuang91/InterCap

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=intercap
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

download() {
  local filepath=$1
  wget --post-data "username=$encoded_email&password=$password" \
    "https://download.is.tue.mpg.de/download.php?domain=intercap&resume=1&sfile=$filepath" \
    -O "$filepath" --no-check-certificate --continue
}

mkdir RGBD_Individuals Res_Individuals

for i in {01..10}; do
  download "RGBD_Individuals/$i.zip"
done

for i in {05..10}; do
  download "Res_Individuals/$i.zip"
done

pushd RGBD_Individuals
for n in *.zip; do extractrm "$n"; done
popd

pushd Res_Individuals
for n in *.zip; do extractrm "$n"; done
popd

# Broken images
echo RGBD_Individuals/07/04/Seg_1/Frames_Cam3/color/00170.jpg > $dataset_dir/ignore_image_paths.txt
echo RGBD_Individuals/09/08/Seg_0/Frames_Cam2/color/00078.jpg >> $dataset_dir/ignore_image_paths.txt
for i in {0..20}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.detect_people --image-root="$dataset_dir" --file-pattern='RGBD_Individuals/**/*.jpg' --batch-size=8  --out-path="$dataset_dir/yolov4_detections.pkl"  --images-per-task=20000
done

python -m posepile.tools.pickle_dict_merge "$dataset_dir"/yolov4_detections_*.pkl "$dataset_dir/yolov4_detections.pkl"
rm "$dataset_dir"/yolov4_detections_*.pkl

python -m posepile.ds.intercap.main --stage=1

for i in {0..5}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/${dataset_name}_downscaled" --out-dir="$DATA_ROOT/${dataset_name}_downscaled/masks"
done

python -m posepile.ds.intercap.main --stage=2