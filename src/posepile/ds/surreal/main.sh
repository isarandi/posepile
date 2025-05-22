#!/usr/bin/env bash
#@INPROCEEDINGS{varol17_surreal,
#  title     = {Learning from Synthetic Humans},
#  author    = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
#  booktitle = {CVPR},
#  year      = {2017}
#}
# https://www.di.ens.fr/willow/research/surreal/data/
# https://github.com/gulvarol/surreal
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/surreal"
# Logging in
echo 'To download the SURREAL dataset, you first need to register on the official website' \
  'at https://www.di.ens.fr/willow/research/surreal/data/'
echo "If that's done, enter your details below:"
read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

wget --http-user="$encoded_email" --http-password="$password" \
  https://lsh.paris.inria.fr/SURREAL/SURREAL_v1.tar.gz
tar xf SURREAL_v1.tar.gz
mv cmu/* ./
rmdir cmu

python -m posepile.ds.surreal.extract_frames

python -m posepile.ds.surreal.main

# MuCo-SURREAL (compositing SURREAL like MuCo-3DHP)
python -m posepile.ds.surreal.muco_surreal --stage=1
python -m humcentr_cli.detect_people \
  --image-root="$DATA_ROOT/muco-surreal2/images" \
  --out-path="$DATA_ROOT/muco-surreal2/yolov4_detections.pkl"
python -m posepile.ds.surreal.muco_surreal --stage=2
