#!/usr/bin/env bash

set -euo pipefail
source posepile/functions.sh
check_data_root

subdir_name=body_models
subdir="$DATA_ROOT/$subdir_name"
mkdircd "$subdir"
mkdir -p smpl smplx smplh smplh16 smplxlh

read -rp 'Email: ' email
read -rsp 'Password: ' password
encoded_email=$(urlencode "$email")

download() {
  local domain=$1
  local filename=$2
  local filepath=$filename
  wget --post-data "username=$encoded_email&password=$password" \
    "https://download.is.tue.mpg.de/download.php?domain=${domain}&resume=1&sfile=${filename}" \
    -O "$filepath" --no-check-certificate --continue
}

download_and_extract() {
  local domain=$1
  local filename=$2
  local filepath=$filename
  download "$domain" "$filename"
  extractrm "$filepath"
}

# SMPL
download_and_extract smpl SMPL_python_v.1.1.0.zip

mv SMPL_python_v.1.1.0/smpl/models/basicmodel_*_lbs_10_207_0_v1.1.0.pkl smpl/
rm -rf SMPL_python_v.1.1.0

pushd smpl
ln -s basicmodel_m_lbs_10_207_0_v1.1.0.pkl SMPL_MALE.pkl
ln -s basicmodel_f_lbs_10_207_0_v1.1.0.pkl SMPL_FEMALE.pkl
ln -s basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl SMPL_NEUTRAL.pkl
popd

wget http://visiondata.cis.upenn.edu/spin/data.tar.gz
extractrm data.tar.gz
mv data/J_regressor_{extra,h36m}.npy smpl/
rm -rf data

# SMPL-X
download_and_extract smplx models_smplx_v1_1.zip
download_and_extract smplx smplx_lockedhead_20230207.zip
download_and_extract smplx model_transfer.zip
download_and_extract smplx smplx_flip_correspondences.zip
download_and_extract smplx smplx_mano_flame_correspondences.zip

mv models/smplx/* smplx/
rm -rf models

mv models_lockedhead/smplx/* smplxlh/
rm -rf models_lockedhead

mv smplx_flip_correspondences.npz smplx/
mv SMPL-X__FLAME_vertex_ids.npy MANO_SMPLX_vertex_ids.pkl smplx/

# Kid templates
download agora smpl_kid_template.npy
mv smpl_kid_template smpl/kid_template.npy
ln -s smpl/kid_template.npy smplh/kid_template.pkl
ln -s smpl/kid_template.npy smplh16/kid_template.pkl

download agora smplx_kid_template.npy
mv smplx_kid_template.npy smplx/kid_template.npy
ln -s smplx/kid_template.npy smplxlh/kid_template.pkl

wget https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_to_J14.pkl -O smplx/SMPLX_to_J14.pkl

# SMPL+H
download_and_extract mano mano_v1_2.zip
mv mano_v1_2/models/SMPLH_* smplh/

download_and_extract mano smplh.tar.xz
mv male female neutral smplh16/

