#!/usr/bin/env bash
#@inproceedings{Black_CVPR_2023,
#  title = {{BEDLAM}: A Synthetic Dataset of Bodies Exhibiting Detailed Lifelike Animated Motion},
#  author = {Black, Michael J. and Patel, Priyanka and Tesch, Joachim and Yang, Jinlong},
#  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
#}
# https://bedlam.is.tue.mpg.de

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=bedlam
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

#wget https://bedlam.is.tuebingen.mpg.de/media/upload/be_imagedata_download.zip
#extractrm be_imagedata_download.zip

bash be_download.sh mp4 masks gt
extractrm *.tar *.tar.gz

# Get the smpl, smplx gt zips, then:
extractrm *.zip

download_and_extract() {
  local filename=$1
  local filepath=training_images/$filename
  wget --post-data "username=$encoded_email&password=$password" \
    "https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=bedlam_images_train/$filename" \
    -O "$filepath" --no-check-certificate --continue
  tar -xf "$filepath" -C "training_images"
  rm "$filepath"
}

mkdir -p training_images
read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")
files=(
       "20221010_3_1000_batch01hand_6fps.tar"
       "20221011_1_250_batch01hand_closeup_suburb_a_6fps.tar"
       "20221011_1_250_batch01hand_closeup_suburb_b_6fps.tar"
       "20221011_1_250_batch01hand_closeup_suburb_c_6fps.tar"
       "20221011_1_250_batch01hand_closeup_suburb_d_6fps.tar"
       "20221012_1_500_batch01hand_closeup_highSchoolGym_6fps.tar"
       "20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps.tar"
       "20221013_3-10_500_batch01hand_static_highSchoolGym_6fps.tar"
       "20221013_3_250_batch01hand_orbit_bigOffice_6fps.tar"
       "20221013_3_250_batch01hand_static_bigOffice_6fps.tar"
       "20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps.tar"
       "20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps.tar"
       "20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps.tar"
       "20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps.tar"
       "20221017_3_1000_batch01hand_6fps.tar"
       "20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps.tar"
       "20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps.tar"
       "20221019_1_250_highbmihand_closeup_suburb_b_6fps.tar"
       "20221019_1_250_highbmihand_closeup_suburb_c_6fps.tar"
       "20221019_3-8_1000_highbmihand_static_suburb_d_6fps.tar"
       "20221019_3_250_highbmihand_6fps.tar"
       "20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.tar"
       "20221022_3_250_batch01handhair_static_bigOffice_30fps.tar"
       "20221024_10_100_batch01handhair_zoom_suburb_d_30fps.tar"
       "20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps.tar")

for file in "${files[@]}"; do
  download_and_extract "$file"
done

mv all_npz_12_training/20221020{-,_}3-8_250_highbmihand_zoom_highSchoolGym_a_6fps.npz

python -m humcentr_cli.detect_people_video --video-dir="$dataset_dir" --output-dir="$dataset_dir/detections" --file-pattern='*.mp4' --videos-per-task=100000000
python -m posepile.ds.bedlam.main