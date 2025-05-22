#!/usr/bin/env bash
#@inproceedings{ben2021ikea,
#  title={The ikea asm dataset: Understanding people assembling furniture through actions, objects and pose},
#  author={Ben-Shabat, Yizhak and Yu, Xin and Saleh, Fatemeh and Campbell, Dylan and Rodriguez-Opazo, Cristian and Li, Hongdong and Gould, Stephen},
#  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
#  pages={847--859},
#  year={2021}
#}
# https://ikeaasm.github.io/
set -euo pipefail
source posepile/functions.sh
check_data_root

# Download data here

mkdircd "$DATA_ROOT/ikea"

for name in *.zip; do
  extractrm "$name"
done

python -m posepile.ds.ikea.main --extract-annotated-frames
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/ikea" --out-path="$DATA_ROOT/ikea/yolov4_detections.pkl"

python -m posepile.ds.ikea.main --stage=1

# num_images=$(find "$DATA_ROOT/ikea_downscaled/" -name '*.jpg' | wc -l)
# should be around 26913
# Then we segment these downscaled images
for i in {0..2}; do
  SLURM_ARRAY_TASK_ID=$i python -m humcentr_cli.segment_people --image-root="$DATA_ROOT/ikea_downscaled" --out-dir="$DATA_ROOT/ikea_downscaled/masks"
done

python -m posepile.ds.ikea.main --stage=2
