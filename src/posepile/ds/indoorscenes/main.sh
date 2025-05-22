#!/usr/bin/env bash
#A. Quattoni, and A.Torralba. Recognizing Indoor Scenes. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.
# https://web.mit.edu/torralba/www/indoor.html

set -euo pipefail
source posepile/functions.sh
check_data_root
dataset_name=indoorscenes
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
wget http://web.mit.edu/torralba/www/TrainImages.txt
wget http://web.mit.edu/torralba/www/TestImages.txt
wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09annotations.tar

extractrm *.tar