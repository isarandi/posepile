#!/usr/bin/env bash
#@inproceedings{vonMarcard2018,
#title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
#author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
#booktitle = {European Conference on Computer Vision (ECCV)},
#year = {2018},
#}
# https://virtualhumans.mpi-inf.mpg.de/3DPW/

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=3dpw
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip
extractrm imageFiles.zip

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip
extractrm sequenceFiles.zip
rm -rf __MACOSX


python -m humcentr_cli.detect_people --image-root="$dataset_dir/imageFiles" --out-path="$dataset_dir/yolov4_detections.pkl"

python -m posepile.ds.tdpw.main


