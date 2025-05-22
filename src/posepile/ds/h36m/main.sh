#!/usr/bin/env bash
# @article{h36m_pami,
#author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu,  Cristian},
#title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
#journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
#publisher = {IEEE Computer Society},
#volume = {36},
#number = {7},
#pages = {1325-1339},
#month = {jul},
#year = {2014}
#}
# @inproceedings{IonescuSminchisescu11,
#author = {Catalin Ionescu, Fuxin Li, Cristian Sminchisescu},
#title = {Latent Structured Models for Human Pose Estimation},
#booktitle = {International Conference on Computer Vision},
#year = {2011}
#}
# http://vision.imar.ro/human3.6m
set -euo pipefail
source posepile/functions.sh
check_data_root

# Logging in
echo 'To download the Human3.6M dataset, you first need to register on the official website at http://vision.imar.ro/human3.6m'
echo "If that's done, enter your details below:"
read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

login_url="https://vision.imar.ro/human3.6m/checklogin.php"
download_url="http://vision.imar.ro/human3.6m/filebrowser.php"
cookie_path=$(mktemp)

_term() {
  # Make sure to clean up the cookie file
  rm "$cookie_path"
}
trap _term SIGTERM SIGINT

curl "$login_url" --insecure --verbose --data "username=$encoded_email&password=$password" --cookie-jar "$cookie_path" --cookie "$cookie_path"

get_file() {
  curl --remote-name --remote-header-name --verbose --cookie-jar "$cookie_path" --cookie "$cookie_path" "$1"
}

get_subject_data() {
  get_file "$download_url?download=1&filepath=Videos&filename=SubjectSpecific_$1.tgz&downloadname=$2"
  get_file "$download_url?download=1&filepath=Segments/mat_gt_bb&filename=SubjectSpecific_$1.tgz&downloadname=$2"
  get_file "$download_url?download=1&filepath=Poses/D3_Positions&filename=SubjectSpecific_$1.tgz&downloadname=$2"
}

mkdircd "$DATA_ROOT/h36m"

get_subject_data 1 S1
get_subject_data 6 S5
get_subject_data 7 S6
get_subject_data 2 S7
get_subject_data 3 S8
get_subject_data 4 S9
get_subject_data 5 S11
get_file http://vision.imar.ro/human3.6m/code-v1.2.zip

for i in 1 5 6 7 8 9 11; do
  tar -xvf "Videos_S$i.tgz"
  rm "Videos_S$i.tgz"
  tar -xvf "Segments_mat_gt_bb_S$i.tgz"
  rm "Segments_mat_gt_bb_S$i.tgz"
  tar -xvf "Poses_D3_Positions_S$i.tgz"
  rm "Poses_D3_Positions_S$i.tgz"
done

extractrm code-v1.2.zip

python -m posepile.ds.h36m.extract_frames_and_boxes
python -m posepile.ds.h36m.main

# num_images=$(find "$DATA_ROOT/h36m_downscaled" -name '*.jpg' | wc -l)
# should be around 173252
# Then we segment these downscaled images
for i in {0..17}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/h36m_downscaled" --out-dir="$DATA_ROOT/h36m_downscaled/masks"
done
