#!/usr/bin/env bash
#Geometric Pose Affordance: 3D Human Pose with Scene Constraints
#Zhe Wang, Liyan Chen, Shuarya Rathore, Daeyun Shin, and Charless Fowlkes.
# https://www.cs.utexas.edu/~liyanc/projects/gpa-dataset/
# https://arxiv.org/abs/1905.07718
set -euo pipefail
source posepile/functions.sh
check_data_root

# Get data from https://drive.google.com/drive/u/2/folders/14SWrgO3d_Ss2Vw3Q_vWb0b6JEZ5g8UgW
mkdircd "$DATA_ROOT/gpa"

tar xf gaussian_fullimg.tar.gz
tar xf img_jpg_new_resnet101deeplabv3humanmask.tar.gz

python -m posepile.gpa.create_directory_structure
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/gpa/Gaussian_img_jpg_new" --out-path="$DATA_ROOT/gpa/yolov4_detections.pkl"
python -m posepile.ds.gpa.main