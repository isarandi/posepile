#!/usr/bin/env bash
# @inproceedings{Zhang:ECCV:2022,
#   title = {EgoBody: Human Body Shape and Motion of Interacting People from Head-Mounted Devices},
#   author = {Zhang, Siwei and Ma, Qianli and Zhang, Yan and Qian, Zhiyin and Kwon, Taein and Pollefeys, Marc and Bogo, Federica and Tang, Siyu},
#   booktitle = {European conference on computer vision (ECCV)},
#   month = oct,
#   year = {2022}
#}
# https://sanweiliti.github.io/egobody/egobody.html
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/egobody"

# Download all files from https://egobody.ethz.ch/data/ into the egobody directory

extractrm annotation_egocentric_smpl_npz.zip calibrations.zip kinect_cam_params.zip smpl*
rsync --info=progress2 -a smplx_*/ smplx/

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/egobody/egocentric_color.zip" --out-path="$DATA_ROOT/egobody/yolov4_detections_ego.pkl" --batch-size=8
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/egobody/kinect_color.zip" --out-path="$DATA_ROOT/egobody/yolov4_detections_kinect.pkl" --batch-size=8
python -m posepile.tools.pickle_dict_merge "$DATA_ROOT/egobody/"/yolov4_detections_*.pkl "$DATA_ROOT/egobody/yolov4_detections.pkl"

python -m posepile.ds.egobody.main