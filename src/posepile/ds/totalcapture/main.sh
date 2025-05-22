#!/usr/bin/env bash
# @inproceedings{Trumble:BMVC:2017,
#	AUTHOR = "Trumble, Matt and Gilbert, Andrew and Malleson, Charles and  Hilton, Adrian and Collomosse, John",
#	TITLE = "Total Capture: 3D Human Pose Estimation Fusing Video and Inertial Sensors",
#	BOOKTITLE = "2017 British Machine Vision Conference (BMVC)",
#	YEAR = "2017"}
# https://cvssp.org/data/totalcapture/
# https://github.com/zhezh/TotalCapture-Toolbox
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/totalcapture"

echo 'To download the TotalCapture dataset, you first need to get permission' \
  'at https://cvssp.org/data/totalcapture/'
echo "If that's done, enter your details below:"
read -rp 'Email: ' username
read -rsp 'Password: ' password

mkdir -p {mattes,video}/s{1..5}
mkdir -p vicon

root_url=https://cvssp.org/data/totalcapture/data/dataset/
getit(){
  wget -O "$1" --http-user="$user" --http-password="$password" "$root_url/$1"
}

for subj in s1 s2 s3; do
  for action in acting freestyle rom walking; do
    for rep in 1 2 3; do
      getit "video/${subj}/${subj}_${action}${rep}.tar.gz"
      getit "mattes/${subj}/${subj}_${action}${rep}.tar.gz"
    done
  done
done

for subj in s4 s5; do
  for action in acting3 freestyle1 freestyle3 rom3 walking2; do
    getit "video/$subj/${subj}_${action}.tar.gz"
    getit "mattes/$subj/${subj}_${action}.tar.gz"
  done
done

for subj in s1 s2 s3 s4 s5; do
  getit "vicon/${subj}_vicon_pos_ori.tar.gz"
done

getit calibration.cal

find . -name '*.tar.gz' | while read name; do
  echo Extracting "$name"...
  (cd "$(dirname "$name")"; tar xf "$(basename "$name")")
done

find . -name '*.tar.gz' -delete

for i in {0..367}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.totalcapture.main --stage=1
done

python -m posepile.ds.totalcapture.main --stage=2