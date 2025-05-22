#!/usr/bin/env bash
# @article{ng2019you2me,
#  title={You2Me: Inferring Body Pose in Egocentric Video via First and Second Person Interactions},
#  author={Ng, Evonne and Xiang, Donglai and Joo, Hanbyul and Grauman, Kristen},
#  journal={CVPR},
#  year={2020}
#}
# https://vision.cs.utexas.edu/projects/you2me/
# https://github.com/facebookresearch/you2me
# https://github.com/facebookresearch/you2me/tree/master/data

set -euo pipefail
source posepile/functions.sh
check_data_root

dataset_name=you2me
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"


wget http://dl.fbaipublicfiles.com/you2me/you2me_ds_release_kinect.tar
wget http://dl.fbaipublicfiles.com/you2me/you2me_ds_release_cmu.tar
extractrm you2me_ds_release_kinect.tar you2me_ds_release_cmu.tar