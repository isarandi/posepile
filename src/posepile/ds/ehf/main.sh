#!/usr/bin/env bash
#@inproceedings{SMPL-X:2019,
#  title = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
#  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
#  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
#  pages     = {10975--10985},
#  year = {2019}
#}
# https://smpl-x.is.tue.mpg.de

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=ehf
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

download_and_extract() {
  local domain=$1
  local filename=$2
  local filepath=$filename
  wget --post-data "username=$encoded_email&password=$password" \
    "https://download.is.tue.mpg.de/download.php?domain=${domain}&resume=1&sfile=${filename}" \
    -O "$filepath" --no-check-certificate --continue
  extractrm $filepath
}

read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")
download_and_extract smplx EHF.zip