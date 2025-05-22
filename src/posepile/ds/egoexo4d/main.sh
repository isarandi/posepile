#!/usr/bin/env bash
#@article{grauman2023ego,
#  title={Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives},
#  author={Grauman, Kristen and Westbury, Andrew and Torresani, Lorenzo and Kitani, Kris and Malik, Jitendra and Afouras, Triantafyllos and Ashutosh, Kumar and Baiyya, Vijay and Bansal, Siddhant and Boote, Bikram and others},
#  journal={arXiv preprint arXiv:2311.18259},
#  year={2023}
#}
# https://ego-exo4d-data.org
# https://arxiv.org/abs/2311.18259

set -euo pipefail
source posepile/functions.sh
check_data_root

dataset_name=egoexo4d
dataset_dir="$DATA_ROOT/$dataset_name"
mkdircd "$dataset_dir"

conda create -n ego4d python=3.11 -y
conda activate ego4d
conda install ego4d

egoexo -o ./ --parts annotations metadata ego_pose_pseudo_gt downscaled_takes/448 -y

python -m posepile.ds.egoexo4d.get_take_uids > takes.txt

bash $CODE_DIR/ds/egoexo4d/download_takes.sh

find -type f -ipath '*/downscaled/448/*' ! -iname '*aria*.mp4' ! -iname '*preview*.mp4' > filtered_videos.txt

find . -type f ! -iname '*aria*.mp4' ! -iname '*preview*.mp4' -ipath '*/downscaled/448/*' | \
while IFS= read -r file; do
    otherfile=$(echo "$file" | sed 's|downscaled/448/||')
    if [ -f "$otherfile" ]; then
      echo "$file";
    fi
done > filtered_videos.txt

python -m humcentr_cli.detect_people_video --video-dir="$dataset_dir" --paths-file="$dataset_dir/filtered_videos.txt" --output-dir="$dataset_dir/detections"

python -m posepile.tools.pickle_dict_merge "$dataset_dir"/yolov4_detections_*.pkl "$dataset_dir/yolov4_detections.pkl"