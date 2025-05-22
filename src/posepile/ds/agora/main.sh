#!/usr/bin/env bash
#@inproceedings{Patel:CVPR:2021,
#  title = {{AGORA}: Avatars in Geography Optimized for Regression Analysis},
#  author = {Patel, Priyanka and Huang, Chun-Hao P. and Tesch, Joachim and Hoffmann, David T. and Tripathi, Shashank and Black, Michael J.},
#  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition ({CVPR})},
#  year = {2021},
#}
# https://agora.is.tue.mpg.de/

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=agora
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

# Logging in
echo 'To download the AGORA dataset, you first need to register on the official website at https://agora.is.tue.mpg.de'
echo "If that's done, enter your details below (or use the raw commands of the script):"
read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

login_url="https://agora.is.tue.mpg.de/login.php"
download_page_url="https://agora.is.tue.mpg.de/download.php"
download_url="https://psfiles.is.tuebingen.mpg.de/downloads/agora"
cookie_path=$(mktemp)
_term() {
  # Make sure to clean up the cookie file
  rm "$cookie_path"
}
trap _term SIGTERM SIGINT

curl "$login_url" --insecure --verbose --data "username=$encoded_email&password=$password" --cookie-jar "$cookie_path" --cookie "$cookie_path"

get_file() {
  curl --remote-name --remote-header-name --verbose -H "Referer: $download_page_url" --cookie-jar "$cookie_path" --cookie "$cookie_path" "$1"
}

for i in {0..9}; do
  get_file "$download_url/train_images_3840x2160_${i}-zip"
done

get_file "$download_url/validation_images_3840x2160-zip"
get_file "$download_url/test_images_3840x2160-zip"
get_file "$download_url/train_masks_3840x2160-zip"
get_file "$download_url/validation_masks_3840x2160-zip"

get_file "$download_url/smplx_gt-zip"
get_file "$download_url/smpl_gt-zip"
get_file "$download_url/gt_scan_info-zip"
get_file "$download_url/smpl_kid_template-npy"
get_file "$download_url/smplx_kid_template-npy"
get_file "$download_url/train_Cam-zip"
get_file "$download_url/train_SMPL-zip"
get_file "$download_url/train_SMPLX-zip"
get_file "$download_url/validation_Cam-zip"
get_file "$download_url/validation_SMPL-zip"
get_file "$download_url/validation_SMPLX-zip"
get_file "$download_url/train_Cam_ReadMe-md"
get_file "$download_url/validation_Cam_ReadMe-md"

wget https://raw.githubusercontent.com/microsoft/AirSim/a7e467ebca707bba5b446836331075e90e3e3ab8/docs/seg_rgbs.txt

extractrm *.zip

for subdir in SMPL SMPLX Cam; do
  mv "$dataset_dir/validation_$subdir/$subdir/"* "$dataset_dir/$subdir/"
  rmdir "$dataset_dir/validation_$subdir/$subdir"
  rmdir "$dataset_dir/validation_$subdir"
done

python -m posepile.ds.agora.extract_masks

python -m humcentr_cli.detect_people --image-root="$dataset_dir" --file-pattern='**/*.png' \
  --out-path="$dataset_dir/yolov4_detections.pkl" --image-type=png

python -m posepile.ds.agora.main