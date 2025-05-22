#!/usr/bin/env bash
#@inproceedings{Fabbri18ECCV,
#  title         = {Learning to Detect and Track Visible and Occluded Body Joints in a Virtual World},
#  author        = {Fabbri, Matteo and Lanzi, Fabio and Calderara, Simone and Palazzi, Andrea and
#                   Vezzani, Roberto and Cucchiara, Rita},
#  booktitle     = ECCV,
#  year          = {2018},
#  doi = {10.1007/978-3-030-01225-0_27}
#}
# https://github.com/fabbrimatteo/JTA-Dataset
set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/jta"

git clone https://github.com/fabbrimatteo/JTA-Dataset.git
python JTA-Dataset/download_data.py


for i in {0..127}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.detect_people_video --video-dir="$DATA_ROOT/jta/videos" --output-dir="$DATA_ROOT/jta/detections" --file-pattern='*.mp4' --videos-per-task=2
done

# no GPU needed
for i in {0..127}; do
  SLURM_ARRAY_TASK_ID=$i python -m posepile.ds.jta.main --stage=1
done

python -m posepile.ds.jta.main --stage=2
