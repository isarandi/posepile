#!/usr/bin/env bash
# L. Sigal, A. Balan and M. J. Black.
# HumanEva: Synchronized Video and Motion Capture Dataset and Baseline Algorithm for Evaluation of Articulated Human Motion,
# In International Journal of Computer Vision, Vol. 87 (1-2), 2010.
# http://humaneva.is.tue.mpg.de/
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/humaneva"

echo 'To download the HumanEva dataset, you first need to register on the official website at http://humaneva.is.tue.mpg.de'
echo "If that's done, enter your details below (or use the raw commands of the script):"
read -rp 'Username: ' username
read -rsp 'Password: ' password

login_url="http://humaneva.is.tue.mpg.de/login"
download_page_url="http://humaneva.is.tue.mpg.de/datasets_human_1"
download_url="http://humaneva.is.tue.mpg.de/main/download"
cookie_path=$(mktemp)
_term() {
  # Make sure to clean up the cookie file
  rm "$cookie_path"
}
trap _term SIGTERM SIGINT

curl "$login_url" --insecure --verbose --data "username=$user&password=$password" --cookie-jar "$cookie_path" --cookie "$cookie_path"

get_file() {
  curl --remote-name --remote-header-name --verbose -H "Referer: $download_page_url" --cookie-jar "$cookie_path" --cookie "$cookie_path" "$1"
}

# TODO complete the download commands here
# http://humaneva.is.tue.mpg.de/main/download?file=HumanEvaI_Data_CD1.tar

python -m posepile.humaneva.extract_frames.py

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/humaneva" --out-path="$DATA_ROOT/humaneva/yolov4_detections.pkl" --file-pattern='**/Image_Data/*_(C?)/*.jpg'
python -m posepile.ds.humaneva.main --stage=1