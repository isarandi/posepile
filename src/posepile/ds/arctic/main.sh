#!/usr/bin/env bash
# @inproceedings{fan2023arctic,
#  title = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
#  author = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
#  booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  year = {2023}
#}
# https://arctic.is.tue.mpg.de/
# https://github.com/zc-alexfan/arctic

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=arctic
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

download_file() {
  local filepath=$1
  mkdir -p "$(dirname "$filepath")"
  wget --post-data "username=$encoded_email&password=$password" \
    "https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/$filepath" \
    -O "$filepath" --no-check-certificate --continue
}

cat "$CODE_DIR/ds/arctic/urls" | while read p; do
  if [[ ! -f "$p" ]]; then
    download_file "$p"
  fi
done

# Only extracting annotations, we will process image data from zip files directly
extractrm *.zip mocap/*.zip

for zippath in $dataset_dir/images_zips/*/*.zip; do
  echo "$zippath"
  outpath="${zippath%.zip}.pkl"
  if [[ -f "$outpath" ]]; then
    continue
  fi

  # check if zippath ends with s01/laptop_use_04.zip
  if [[ $zippath =~ s01/laptop_use_04.zip ]]; then
    python -m humcentr_cli.detect_people --image-root="$zippath" --out-path="$outpath" --batch-size=8 --ignore-paths-file=<(echo 7/00737.jpg)
  elif [[ $zippath =~ s08/mixer_use_02.zip ]]; then
    python -m humcentr_cli.detect_people --image-root="$zippath" --out-path="$outpath" --batch-size=8 --ignore-paths-file=<(echo 3/00295.jpg)
  else
    python -m humcentr_cli.detect_people --image-root="$zippath" --out-path="$outpath" --batch-size=8
  fi
done

python -m posepile.ds.arctic.main --stage=1

for i in {0..10}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/${dataset_name}_downscaled" --out-dir="$DATA_ROOT/${dataset_name}_downscaled/masks"
done

python -m posepile.ds.arctic.main --stage=2